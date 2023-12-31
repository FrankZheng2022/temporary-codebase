# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
import io

def task_names(split=1):
    if split == 1:
        task_names = ['assembly', 'basketball', 'button-press-topdown', 'button-press-topdown-wall', 'button-press', 'button-press-wall', 'coffee-button', 'coffee-pull', 'coffee-push', 'dial-turn', 'disassemble', 'door-close', 'door-open', 'drawer-close', 'drawer-open', 'faucet-open', 'faucet-close', 'hammer', 'handle-press-side', 'handle-press', 'handle-pull-side', 'handle-pull', 'lever-pull', 'peg-insert-side', 'pick-place-wall', 'pick-out-of-hole', 'reach', 'push-back', 'push', 'pick-place', 'plate-slide', 'plate-slide-side', 'plate-slide-back', 'plate-slide-back-side', 'peg-unplug-side', 'soccer', 'stick-push', 'stick-pull', 'push-wall', 'reach-wall', 'shelf-place', 'sweep-into', 'sweep', 'window-open', 'window-close']
    elif split == 2:
        ###  ['pick-place-wall', 'reach-wall', 'button-press', 'coffee-button', 'button-press-topdown']
        task_names = ['assembly', 'basketball', 'button-press-topdown-wall', 'button-press-wall', 'coffee-pull', 'coffee-push', 'dial-turn', 'disassemble', 'door-close', 'door-open', 'drawer-close', 'drawer-open', 'faucet-open', 'faucet-close', 'hammer', 'handle-press-side', 'handle-press', 'handle-pull-side', 'handle-pull', 'lever-pull', 'peg-insert-side', 'pick-out-of-hole', 'reach', 'push-back', 'push', 'pick-place', 'plate-slide', 'plate-slide-side', 'plate-slide-back', 'plate-slide-back-side', 'peg-unplug-side', 'soccer', 'stick-push', 'stick-pull', 'push-wall', 'shelf-place', 'sweep-into', 'sweep', 'window-open', 'window-close', 'hand-insert', 'door-unlock', 'door-lock', 'box-close', 'bin-picking']
    elif split == 4:
        ### ['drawer-open', 'pick-place', 'button-press-wall', 'assembly', 'door-open']
        task_names = ['basketball', 'button-press-topdown', 'button-press-topdown-wall', 'button-press', 'coffee-button', 'coffee-pull', 'coffee-push', 'dial-turn', 'disassemble', 'door-close', 'drawer-close', 'faucet-open', 'faucet-close', 'hammer', 'handle-press-side', 'handle-press', 'handle-pull-side', 'handle-pull', 'lever-pull', 'peg-insert-side', 'pick-place-wall', 'pick-out-of-hole', 'reach', 'push-back', 'push', 'plate-slide', 'plate-slide-side', 'plate-slide-back', 'plate-slide-back-side', 'peg-unplug-side', 'soccer', 'stick-push', 'stick-pull', 'push-wall', 'reach-wall', 'shelf-place', 'sweep-into', 'sweep', 'window-open', 'window-close', 'hand-insert', 'door-unlock', 'door-lock', 'box-close', 'bin-picking']
        
    return task_names

### code_pred: code sequence predicted by the meta-policy
### code_target: target code sequence provided by the tokenizer
### code_dist: pre-computed distance matrix between codes
def cal_tok_dist(code_pred, code_actual, code_dist):
    len_match = min(len(code_pred), len(code_actual))
    dist  = 0.
    for i in range(len_match):
        dist += code_dist[code_pred[i]][code_actual[i]]
    if len(code_pred) < len(code_actual):
        dist += np.max(code_dist) * (len(code_actual) - len(code_pred))
    return dist

def tokenize_vocab(traj_tok, vocab_lookup, merges):
    traj_vocab = []
    for i in range(len(traj_tok)):
        tok = traj_tok[i]
        idx = i
        while idx < len(traj_tok) - 1 and (tok, traj_tok[idx+1]) in merges:
            tok += traj_tok[idx+1]
        traj_vocab.append(vocab_lookup[tok])
    return traj_vocab

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())

def forward_chunk(tensor, fn, chunk=3):
    batch_size = tensor.shape[0]
    chunk_size = batch_size // chunk
    output_batches = None
    tensor_chunks = []
    for i in range(chunk):
        tensor_chunk = tensor[i*chunk_size:(i+1)*chunk_size]
        tensor_chunks.append(tensor_chunk)    
    del tensor
    torch.cuda.empty_cache()
    
    for i in range(chunk):
        output_batch = fn(tensor_chunks[i])
        tensor_chunks[i] = None
        torch.cuda.empty_cache()
        output_size  = len(output_batch)
        if output_batches is None:
            output_batches = [[] for i in range(output_size)]
        for j in range(output_size):
            output_batches[j].append(output_batch[j])
    return [torch.concat(item) for item in output_batches]
        
    

def spr_loss(f_x1s, f_x2s):
    f_x1 = F.normalize(f_x1s, p=2., dim=-1, eps=1e-3)
    f_x2 = F.normalize(f_x2s, p=2., dim=-1, eps=1e-3)
    loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
    return loss
    
class AllGatherFunc(Function):
    @staticmethod
    def forward(ctx, input):
        # Create output tensor
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Sum up the gradients from all outputs
        grad_input = sum(grad_outputs)
        return grad_input

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def set_requires_grad(net, value=False):
    """Enable/disable gradients for a given (sub)network."""
    for param in net.parameters():
        param.requires_grad_(value)

### input shape: (batch_size, length, action_dim)
### output shape: (batch_size, action_dim)
class ActionEncoding(nn.Module):
    def __init__(self, action_dim, latent_action_dim, multistep):
        super().__init__()
        self.action_dim = action_dim
        self.action_tokenizer = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.LayerNorm(64), 
            nn.Tanh(), 
            nn.Linear(64, latent_action_dim)
        )
        self.action_seq_tokenizer = nn.Sequential(
            nn.Linear(latent_action_dim*multistep, latent_action_dim*multistep),
            nn.LayerNorm(latent_action_dim*multistep), nn.Tanh()
        )
        self.apply(weight_init)
        
    def forward(self, action, seq=False):
        if seq:
            batch_size = action.shape[0]
            action = self.action_tokenizer(action) #(batch_size, length, action_dim)
            action = action.reshape(batch_size, -1)
            return self.action_seq_tokenizer(action)
        else:
            return self.action_tokenizer(action)



class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def choose(traj_list, max_traj):
    # NOTE: this assumes that random's seed has been set.
    random.shuffle(traj_list)
    return (traj_list if max_traj < 0 else traj_list[:max_traj])


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def expectile_loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

        
def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)

def to_torch_distribute(xs):
    return tuple(torch.as_tensor(x).cuda() for x in xs)

def encode_multiple(encoder, xs, detach_lst):
    length = [x.shape[0] for x in xs]
    xs, xs_lst = torch.cat(xs, dim=0), []
    xs = encoder(xs)
    start = 0
    for i in range(len(detach_lst)):
        x = xs[start:start+length[i], :]
        if detach_lst[i]:
            x = x.detach()
        xs_lst.append(x)
        start += length[i]
    return xs_lst
    
    

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


# class ActionQuantizer(nn.Module):
#     def __init__(self, agent):
#         super().__init__()
#         self.agent = agent
#         set_requires_grad(self.agent, False)
        
#     def forward(self, action):
#         action_en = self.agent.TACO.proj_aseq(action)
#         _, _, _, _, min_encoding_indices = self.agent.TACO.quantizer(action_en)
#         return min_encoding_indices
        