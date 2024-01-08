# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import torchvision.transforms as T
from quantizer import VectorQuantizer
import time
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from resnet18_encoder import ResnetEncoder
from policy_head import GMMHead


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, input_size, inv_freq_factor=10, factor_ratio=None):
        super().__init__()
        self.input_size = input_size
        self.inv_freq_factor = inv_freq_factor
        channels = self.input_size
        channels = int(np.ceil(channels / 2) * 2)

        inv_freq = 1.0 / (
            self.inv_freq_factor ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.channels = channels
        self.register_buffer("inv_freq", inv_freq)

        if factor_ratio is None:
            self.factor = 1.0
        else:
            factor = nn.Parameter(torch.ones(1) * factor_ratio)
            self.register_parameter("factor", factor)

    def forward(self, x):
        pos_x = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x * self.factor

    def output_shape(self, input_shape):
        return input_shape

    def output_size(self, input_size):
        return input_size
    
    
class RelativePositionalEncoder(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0.0, n_embd, 2.0) / n_embd))  # [NS/2]
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        pos_seq = torch.arange(x.shape[1] - 1, -1, -1.0)
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)  # [N, NS/2] <- outer product
        pos_enc = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)  # [N, NS]
        return pos_enc.to(x.device)

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class ActionEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, action_dim, obs_dependent):
        super().__init__()
        self.a_embedding = nn.Sequential(
            nn.Linear(action_dim, 64),
            #nn.LayerNorm(64), 
            nn.Tanh(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, feature_dim),
            nn.Tanh(),
        )
        self.sa_embedding = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.obs_dependent = obs_dependent
        self.apply(utils.weight_init)
    
    def forward(self, z, a):
        u = self.a_embedding(a)
        if self.obs_dependent:
            return self.sa_embedding(z.detach()+u)
        else:
            return u


class TACO(nn.Module):
    """
    Constrastive loss
    """

    def __init__(self, feature_dim, action_dim, hidden_dim, encoder, nstep, nstep_history, n_code, vocab_size, device, obs_dependent, decoder_type):
        super(TACO, self).__init__()

        self.nstep = nstep
        self.encoder = encoder
        self.device = device
        
        self.transition = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.action_encoder = ActionEncoder(feature_dim, hidden_dim, action_dim, obs_dependent)
        
        self.a_quantizer = VectorQuantizer(n_code, feature_dim)
                            
        if decoder_type== 'deterministic':
            self.decoder = nn.Sequential(
                nn.Linear(feature_dim*4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        elif decoder_type == 'gmm':
            self.decoder =  GMMHead(feature_dim*4, action_dim, hidden_size=hidden_dim)
        else:
            print('Decoder type not supported!')
            raise Exception
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim*4, nhead=4, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_embedding  = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.proj_s = nn.Sequential(nn.Linear(feature_dim, feature_dim),
                                    nn.ReLU(),
                                    nn.Linear(feature_dim, feature_dim))
        
        self.apply(utils.weight_init)
    
    def encode(self, x, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z = self.encoder(x)
                z_out = self.proj_s(z)
        else:
            z = self.encoder(x)
            z_out = self.proj_s(z)
        return z,  z_out


class TACORepresentation:
    def __init__(self, obs_shape, action_dim, device, lr, feature_dim,
                 hidden_dim, nstep, nstep_history, trunk, pcgrad, n_code, vocab_size, 
                 obs_dependent, alpha, decoder_type, positional_embedding):
        self.device = device
        self.nstep = nstep
        self.pcgrad = pcgrad
        self.scaler = GradScaler()
        self.feature_dim = feature_dim
        self.n_code = n_code
        self.alpha  = alpha
        self.decoder_type = decoder_type
        self.positional_embedding = positional_embedding
        self.encoders = torch.nn.ModuleList([ResnetEncoder(input_shape=obs_shape, output_size=feature_dim).to(device),
                                             ResnetEncoder(input_shape=obs_shape, output_size=feature_dim).to(device),
                                             nn.Sequential(
                                                nn.Linear(9, hidden_dim),
                                                nn.ReLU(),
                                                nn.Linear(hidden_dim, feature_dim),
                                            ),
                                            nn.Sequential(
                                                nn.Linear(768, hidden_dim),
                                                nn.ReLU(),
                                                nn.Linear(hidden_dim, feature_dim),
                                            ),
                                            ])
                                    
        
        self.TACO = DDP(TACO(feature_dim, action_dim, hidden_dim, self.encoders, nstep, nstep_history, n_code, vocab_size, device, obs_dependent, decoder_type).to(device))
        if positional_embedding == 'absolute':
            self.positional_embedding = SinusoidalPositionEncoding(feature_dim*4)
        elif positional_embedding == 'relative':
            self.positional_embedding = RelativePositionalEncoder(feature_dim*4)
        else:
            print('Positional Embedding Not Implemented')
        
        self.taco_opt = torch.optim.Adam(self.TACO.parameters(), lr=lr)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # data augmentation
        self.aug = RandomShiftsAug(pad=6)
        
        self.train()

    def train(self, training=True):
        self.training = training
        self.encoders.train(training)
        self.TACO.train(training)
    
    ### Compute an N by N matrix (each index corresponding)
    ### to the distance between two codes
    def cal_distance(self, o_embed):
        ### For each code, decode it back to the raw action
        decode_action_lst = []
        for i in range(self.n_code):
            learned_code   = self.TACO.module.a_quantizer.embedding.weight
            u_quantized    = learned_code[i, :]
            decode_action = self.TACO.module.decoder(o_embed + u_quantized)
            decode_action_lst.append(decode_action)      
        ### Calculate the distance
        code_distance = np.zeros((self.n_code, self.n_code))
        for i in range(self.n_code):
            for j in range(self.n_code):
                decode_action_i = decode_action_lst[i]
                decode_action_j = decode_action_lst[j]
                code_distance[i][j] = F.l1_loss(decode_action_i, decode_action_j).item()
        return code_distance
            
    
    def encode_obs(self, obs_agent, obs_wrist, state, task_embedding, add_task_embedding=True, aug=True):
        if aug:
            z_agent = self.encoders[0](self.aug(obs_agent.float()), langs=task_embedding)
            z_wrist = self.encoders[1](self.aug(obs_wrist.float()), langs=task_embedding)
        else:
            z_agent = self.encoders[0](obs_agent.float(), langs=task_embedding)
            z_wrist = self.encoders[1](obs_wrist.float(), langs=task_embedding)
        state   = self.encoders[2](state.float())
        if add_task_embedding:
            task_embedding = self.encoders[3](task_embedding)
            z = torch.concatenate([z_agent, z_wrist, state, task_embedding], dim=-1)
        else:
            z = torch.concatenate([z_agent, z_wrist, state], dim=-1)
        return z
        
    
    def update_taco(self, obs, action_seq, next_obs_lst):
        metrics = dict()
        index = 0
            
        obs = torch.torch.as_tensor(obs, device=self.device)
        next_obses = [torch.torch.as_tensor(next_obs, device=self.device) for next_obs in next_obs_lst]
        action_seq = [torch.torch.as_tensor(a, device=self.device) for a in action_seq]
        
        o_embed, _ = self.TACO.module.encode(self.aug(obs.float()))
        z = o_embed
        spr_loss, quantize_loss, decoder_loss = 0, 0, 0
        for k in range(self.nstep):
            u = self.TACO.module.action_encoder(o_embed, action_seq[k])
            q_loss, u_quantized, _, _, min_encoding_indices = self.TACO.module.a_quantizer(u)
            quantize_loss += q_loss
            
            ### Decoder Loss
            decode_action = self.TACO.module.decoder(o_embed + u_quantized)
            #decode_action = self.TACO.module.decoder(z + u_quantized)
            d_loss = F.l1_loss(decode_action, action_seq[k])
            decoder_loss += d_loss
        
            ### SPR Loss
            z = self.TACO.module.transition(z+u_quantized)
            next_obs = self.aug(next_obses[k].float())
            o_embed, y_next = self.TACO.module.encode(next_obs, ema=True)
            y_pred = self.TACO.module.predictor(self.TACO.module.proj_s(z)) 
            spr_loss += utils.spr_loss(y_pred, y_next)
        
        self.taco_opt.zero_grad()
        (spr_loss + decoder_loss + quantize_loss).backward()
        self.taco_opt.step()
        metrics['spr_loss']      = spr_loss.item()
        metrics['quantize_loss'] = quantize_loss.item()
        metrics['decoder_loss']  = decoder_loss.item()
        return metrics
        
    
    def update(self, replay_iter, step):
        metrics = dict()
        batch = next(replay_iter)
        obs, _, action_seq, _, _, _, next_obs_lst = batch
        metrics.update(self.update_taco(obs, action_seq, next_obs_lst))
        return metrics
    
    def update_metapolicy(self, replay_iter, step, tok_to_code, 
                          tok_to_idx, idx_to_tok, idx_distance, 
                          cross_entropy=False):
        metrics = dict()
        batch = next(replay_iter)
        obs, action, tok, action_seq, obs_seq = batch
        obs = torch.torch.as_tensor(obs, device=self.device)
        action = torch.torch.as_tensor(action, device=self.device)
        action_seq = [torch.torch.as_tensor(action, device=self.device) for action in action_seq]
        obs_seq = [torch.torch.as_tensor(o, device=self.device) for o in obs_seq]
        obs_seq = [obs] + obs_seq[:-1]
        z_seq   = [self.encoder(o.float()) for o in obs_seq]
        
        tok = torch.torch.as_tensor(tok, device=self.device).reshape(-1)
        z = self.encoder(obs.float())
        with torch.no_grad():
            u = self.TACO.module.action_encoder(z, action)
            _, u_quantized, _, _, min_encoding_indices = self.TACO.module.a_quantizer(u)
            code  = [tok_to_code(x)[0] for x in tok]
            index = torch.tensor([tok_to_idx(x) for x in tok]).long().to(self.device)
            u_quantized = self.TACO.module.a_quantizer.embedding.weight[code, :]
        
        meta_action = self.TACO.module.meta_policy(z.detach())
        if cross_entropy:
            meta_policy_loss = F.cross_entropy(meta_action, index)
        else:
            #idx_distance = torch.ones_like(idx_distance).to(self.device)-torch.eye(idx_distance.shape[0]).to(self.device)
            meta_action_dist = F.gumbel_softmax(meta_action)
            tok_distance = idx_distance[index] ### distance to the target index
            meta_policy_loss = torch.mean(torch.sum(meta_action_dist*tok_distance, dim=-1))
            
        
        ### Iterate over every token and calculate the deecoder l1 loss (of the first action)
        decoder_loss_lst = []
        
        for idx in range(idx_distance.shape[0]):
            
            token_length   = len(tok_to_code(torch.tensor(idx_to_tok[idx])))
            rollout_length = min(len(action_seq), token_length)
            action = torch.concatenate(action_seq[:rollout_length], dim=0)
            z = torch.concatenate(z_seq[:rollout_length], dim=0)
            
            ### Concatenate the codes from step 1 to #rollout_length
            u_quantized_lst = []
            for t in range(rollout_length):
                with torch.no_grad():
                    learned_code   = self.TACO.module.a_quantizer.embedding.weight
                    u_quantized    = learned_code[tok_to_code(torch.tensor(idx_to_tok[idx]))[t], :]
                    u_quantized    = u_quantized.repeat(obs.shape[0], 1)
                u_quantized_lst.append(u_quantized)
            u_quantized = torch.concatenate(u_quantized_lst,dim=0)
            
            ### Decode the codes into action sequences and calculate L1 loss
            decode_action = self.TACO.module.decoder((z + u_quantized).detach())
            decoder_loss = torch.sum(torch.abs(decode_action-action), dim=-1, keepdim=True)
            decoder_loss = torch.sum(decoder_loss.reshape(rollout_length, obs.shape[0]),dim=0).unsqueeze(-1)
            decoder_loss_lst.append(decoder_loss)
        
        ### Shape: (Batch_size, num_indices)
        decoder_loss = torch.cat(decoder_loss_lst, dim=-1)
        meta_action_dist = F.gumbel_softmax(meta_action)
        #meta_action_dist = F.softmax(meta_action)
        decoder_loss = torch.mean(torch.sum(decoder_loss*meta_action_dist, dim=-1))
        
        self.taco_opt.zero_grad()
        (meta_policy_loss+self.alpha*decoder_loss).backward()
        self.taco_opt.step()
        
        metrics['meta_policy_loss'] = meta_policy_loss
        metrics['decoder_loss'] = decoder_loss
        
        return metrics
    
    def update_metapolicy_nonbpe(self, replay_iter, step):
        
        metrics = dict()
        batch  = next(replay_iter)
        obs, action, code, action_seq, obs_lst = batch
        obs    = torch.torch.as_tensor(obs, device=self.device)
        action = torch.torch.as_tensor(action, device=self.device)
        code   = torch.torch.as_tensor(code, device=self.device).reshape(-1)
        z      = self.encoder(obs.float())
        with torch.no_grad():
            u = self.TACO.module.action_encoder(z, action)
            _, u_quantized, _, _, min_encoding_indices = self.TACO.module.a_quantizer(u)
            u_quantized = self.TACO.module.a_quantizer.embedding.weight[code, :]
        
        meta_action = self.TACO.module.meta_policy(z.detach())
        meta_policy_loss = F.cross_entropy(meta_action, code)
    
        decode_action = self.TACO.module.decoder((z + u_quantized).detach())
        decoder_loss = F.l1_loss(decode_action, action)
        
        self.taco_opt.zero_grad()
        (meta_policy_loss+decoder_loss).backward()
        self.taco_opt.step()
        
        metrics['meta_policy_loss'] = meta_policy_loss
        metrics['decoder_loss'] = decoder_loss
        
        return metrics
    
    ### train bc
    def update_bc(self, replay_iter, step):
        metrics = dict()
        batch = next(replay_iter)
        task_embedding, obs_agent, obs_wrist, state, action, _, _, obs_history, pad_idx, pad_mask = batch
        pad_idx = torch.torch.as_tensor(pad_idx, device=self.device)  ### integers (batch_size, )
        pad_mask = torch.torch.as_tensor(pad_mask, device=self.device) ### boolean (batch_size, timestep)
        
        
        obs_agent_history, obs_wrist_history, state_history, task_embedding_history = obs_history
        batch_size, time_step = obs_agent_history.shape[0], obs_agent_history.shape[1]
        ### shape: (batch_size, timestep, 3, 128, 128)
        task_embedding_history = torch.torch.as_tensor(task_embedding_history, device=self.device)
        task_embedding_history = task_embedding_history.reshape(-1, task_embedding_history.shape[-1])
        
        obs_agent_history = torch.torch.as_tensor(obs_agent_history, device=self.device)
        obs_agent_history = obs_agent_history.reshape(-1, 3, 128, 128)
        z_agent_history   = self.encoders[0](self.aug(obs_agent_history.float()), langs=task_embedding_history)
        
        obs_wrist_history = torch.torch.as_tensor(obs_wrist_history, device=self.device)
        obs_wrist_history = obs_wrist_history.reshape(-1, 3, 128, 128)
        z_wrist_history   = self.encoders[1](self.aug(obs_wrist_history.float()), langs=task_embedding_history)
        
        state_history     = torch.torch.as_tensor(state_history, device=self.device)
        state_history     = state_history.reshape(-1, state_history.shape[-1])
        z_state_history   = self.encoders[2](state_history.float())
        
        z_task_embedding  = self.encoders[3](task_embedding_history)
        
        z_agent_history  = z_agent_history.reshape(batch_size, time_step, -1)
        z_wrist_history  = z_wrist_history.reshape(batch_size, time_step, -1)
        z_state_history  = z_state_history.reshape(batch_size, time_step, -1)
        z_task_embedding = z_task_embedding.reshape(batch_size, time_step, -1)
        
        z_history = torch.concatenate([z_agent_history, z_wrist_history, z_state_history, z_task_embedding], dim=-1)
        
        positional_embedding = self.positional_embedding(z_history)
        z_history += positional_embedding.to(self.device)
        mask = nn.Transformer.generate_square_subsequent_mask(z_history.shape[1]).to(self.device)
        z_history = self.TACO.module.transformer_embedding(z_history, mask=mask, src_key_padding_mask=pad_mask)
        pad_idx = pad_idx.unsqueeze(-1).unsqueeze(-1)
        pad_idx = pad_idx.expand(-1, -1, z_history.shape[2])
        z_history = torch.gather(z_history, 1, pad_idx).squeeze(1)
        
        decode_action = self.TACO.module.decoder(z_history)
        action = torch.torch.as_tensor(action, device=self.device)
        if self.decoder_type == 'deterministic':
            bc_loss = F.l1_loss(decode_action, action)
        elif self.decoder_type == 'gmm':
            bc_loss = self.TACO.module.decoder.loss_fn(decode_action, action)
        else:
            print('Decoder type not supported!')
            raise Exception
        
        self.taco_opt.zero_grad()
        bc_loss.backward()
        self.taco_opt.step()
        metrics['bc_loss']  = bc_loss.item()
        return metrics
