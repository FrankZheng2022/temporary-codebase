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
    

class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, trunk=True):
        super().__init__()

        assert len(obs_shape) == 3
        repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                 nn.ReLU())
        # if trunk:
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        self.repr_dim = feature_dim
        # else:
        #     self.trunk = nn.Identity()
        
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return self.trunk(h)
    

class TACO(nn.Module):
    """
    Constrastive loss
    """

    def __init__(self, repr_dim, feature_dim, action_dim, hidden_dim, encoder, nstep, n_code, vocab_size, device, obs_dependent):
        super(TACO, self).__init__()

        self.nstep = nstep
        self.encoder = encoder
        self.device = device
        
        self.meta_policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        
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
        # self.a_quantizer = VectorQuant(
        #     feature_size=feature_dim,    # feature dimension corresponding to the vectors
        #     num_codes=n_code,       # number of codebook vectors
        #     beta=0.98,          # (default: 0.95) commitment trade-off
        #     kmeans_init=False,   # (default: False) whether to use kmeans++ init
        #     norm=None,          # (default: None) normalization for input vector
        #     cb_norm=None,       # (default: None) normalization for codebook vectors
        #     affine_lr=10.0,     # (default: 0.0) lr scale for affine parameters
        #     sync_nu=1.0,        # (default: 0.0) codebook synchronization contribution
        #     replace_freq=20,    # (default: 0) frequency to replace dead codes
        #     dim=-1              # (default: -1) dimension to be quantized
        #         )
                            
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )
        
        
        self.proj_s = nn.Sequential(nn.Linear(repr_dim, feature_dim),
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
                 hidden_dim, nstep, spr, trunk, pcgrad, n_code, vocab_size, obs_dependent):
        self.device = device
        self.nstep = nstep
        self.spr = spr
        self.pcgrad = pcgrad
        self.scaler = GradScaler()
        self.feature_dim = feature_dim
        
        self.encoder = Encoder(obs_shape, feature_dim, 
                                trunk=trunk).to(device)
        
        self.TACO = DDP(TACO(self.encoder.repr_dim, feature_dim, action_dim, hidden_dim, self.encoder, nstep, n_code, vocab_size, device, obs_dependent).to(device))
        
        self.taco_opt = torch.optim.Adam(self.TACO.parameters(), lr=lr)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.TACO.train(training)
    
    
    def update_taco(self, obs, action_seq, next_obs_lst):
        metrics = dict()
        index = 0
            
        obs = torch.torch.as_tensor(obs, device=self.device)
        next_obses = [torch.torch.as_tensor(next_obs, device=self.device) for next_obs in next_obs_lst]
        action_seq = [torch.torch.as_tensor(a, device=self.device) for a in action_seq]
        
        o_embed, _ = self.TACO.module.encode(self.aug(obs.float()))
        z = o_embed
        spr_loss, quantize_loss, decoder_loss, meta_policy_loss = 0, 0, 0, 0
        for k in range(self.nstep):
            u = self.TACO.module.action_encoder(o_embed, action_seq[k])
            q_loss, u_quantized, _, _, min_encoding_indices = self.TACO.module.a_quantizer(u)
            quantize_loss += q_loss
            
            ### Decoder Loss
            #decode_action = self.TACO.module.decoder(o_embed + u_quantized)
            decode_action = self.TACO.module.decoder(z + u_quantized)
            d_loss = F.l1_loss(decode_action, action_seq[k])
            decoder_loss += d_loss
            
            ### Meta policy loss
            meta_action = self.TACO.module.meta_policy(o_embed.detach())
            meta_policy_loss += F.cross_entropy(meta_action, min_encoding_indices.reshape(-1))
        
            ### SPR Loss
            z = self.TACO.module.transition(z+u_quantized)
            next_obs = self.aug(next_obses[k].float())
            o_embed, y_next = self.TACO.module.encode(next_obs, ema=True)
            y_pred = self.TACO.module.predictor(self.TACO.module.proj_s(z)) 
            spr_loss += utils.spr_loss(y_pred, y_next)
        
        self.taco_opt.zero_grad()
        (spr_loss + meta_policy_loss + decoder_loss + quantize_loss).backward()
        self.taco_opt.step()
        metrics['spr_loss']      = spr_loss.item()
        metrics['quantize_loss'] = quantize_loss.item()
        metrics['decoder_loss']  = decoder_loss.item()
        metrics['meta_policy_loss']  = meta_policy_loss.item()
        return metrics
        
    
    def update(self, replay_iter, step):
        metrics = dict()
        batch = next(replay_iter)
        obs, _, action_seq, _, _, _, next_obs_lst = batch
        metrics.update(self.update_taco(obs, action_seq, next_obs_lst))
        return metrics
    
    def update_metapolicy(self, replay_iter, step, tok_to_code, tok_to_idx):
        metrics = dict()
        batch = next(replay_iter)
        obs, action, tok, _, _, _, _ = batch
        obs = torch.torch.as_tensor(obs, device=self.device)
        action = torch.torch.as_tensor(action, device=self.device)
        tok = torch.torch.as_tensor(tok, device=self.device).reshape(-1)
        z = self.encoder(self.aug(obs.float()))
        with torch.no_grad():
            u = self.TACO.module.action_encoder(z, action)
            _, u_quantized, _, _, min_encoding_indices = self.TACO.module.a_quantizer(u)
            code  = [tok_to_code(x) for x in tok]
            index = torch.tensor([tok_to_idx(x) for x in tok]).long().to(self.device)
            u_quantized = self.TACO.module.a_quantizer.embedding.weight[code, :]
            
        meta_action = self.TACO.module.meta_policy(z.detach())
        meta_policy_loss = F.cross_entropy(meta_action, index)

        decode_action = self.TACO.module.decoder((z + u_quantized).detach())
        decoder_loss = F.l1_loss(decode_action, action)
        
        self.taco_opt.zero_grad()
        (meta_policy_loss+decoder_loss).backward()
        self.taco_opt.step()
        
        metrics['meta_policy_loss'] = meta_policy_loss
        metrics['decoder_loss'] = decoder_loss
        
        return metrics

    ### train multitask bc
    def update_multitask_bc(self, replay_iter, step):
        metrics = dict()
        batch = next(replay_iter)
        obs, action, _, _, _, _, _ = batch
            
        obs = torch.torch.as_tensor(obs, device=self.device)
        action = torch.torch.as_tensor(action, device=self.device)
        decode_action = self.TACO.module.decoder(self.TACO.module.encode(obs.float())[0])
        
        decoder_loss = F.l1_loss(decode_action, action)
        
        self.taco_opt.zero_grad()
        decoder_loss.backward()
        self.taco_opt.step()
        metrics['decoder_loss']  = decoder_loss.item()
        return metrics