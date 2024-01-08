# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import traceback
import utils
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from torch.utils.data.distributed import DistributedSampler


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_traj_per_task, max_size, num_workers, nstep,
                 nstep_history, discount, fetch_every, save_snapshot,
                 rank=None, world_size=None,
                 n_code=None, vocab_size=None,
                 min_frequency=None, max_token_length=None):
        self._replay_dir = replay_dir if type(replay_dir) == list else [replay_dir]
        self._max_traj_per_task = max_traj_per_task
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._nstep_history = nstep_history
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self.rank = rank
        self.world_size = world_size
        self.vocab_size = vocab_size
        self.n_code    = n_code
        self.min_frequency = min_frequency
        self.max_token_length = max_token_length
        print('Loading Data into CPU Memory')
        self._preload()

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def __len__(self):
        return self._size

    def _store_episode(self, eps_fn):
        episode = load_episode(eps_fn)
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            # early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        # if not self._save_snapshot:
        #     eps_fn.unlink(missing_ok=True)
        return True

    def _preload(self):
        eps_fns = []
        for replay_dir in self._replay_dir:
            eps_fns.extend(utils.choose(sorted(replay_dir.glob('*.npz'), reverse=True), self._max_traj_per_task))
        if len(eps_fns)==0:
            raise ValueError('No episodes found in {}'.format(self._replay_dir))
        for eps_idx, eps_fn in enumerate(eps_fns):
            if self.rank is not None and eps_idx % self.world_size != self.rank:
                continue
            else:
                self._store_episode(eps_fn)
    
    
    ### ['observation', 'observation_wrist', 'state', 'task_embedding', 'action', 'reward', 'discount']
    def _sample(self):
        # try:
        #     self._try_fetch()
        # except:
        #     traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        task_embedding = episode['task_embedding']
        
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        
        obs_agent = episode['observation'][idx - 1]
        obs_wrist = episode['observation_wrist'][idx - 1]
        state     = episode['state'][idx - 1]
        
        action     = episode['action'][idx]
        action_seq = [episode['action'][idx+i] for i in range(self._nstep)]
        
        next_obs_lst = []
        for i in range(self._nstep):
            obs = (episode['observation'][idx + i], 
                   episode['observation_wrist'][idx + i],
                   episode['state'][idx + i])
            next_obs_lst.append(obs)
            
            
        obs_agent_history, obs_wrist_history, state_history, task_embedding_history = [], [], [], []
        timestep = idx - 1
        
        ### (o_{t-3}, o_{t-2}, o_{t-1}, o_{t}, 0, 0 ...)
        while timestep >= 0 and len(obs_agent_history)<self._nstep_history:
            obs_agent_history = [episode['observation'][timestep][None,:]] + obs_agent_history
            obs_wrist_history = [episode['observation_wrist'][timestep][None,:]] + obs_wrist_history
            state_history     = [episode['state'][timestep][None, :]] + state_history 
            task_embedding_history = [episode['task_embedding'][None, :]] + task_embedding_history
            timestep -= 1
        pad_idx = len(obs_agent_history) - 1
        pad_mask = np.array([False]*(pad_idx+1)+[True]*(self._nstep_history-pad_idx-1))
        
        for i in range(len(obs_agent_history), self._nstep_history):
            obs_agent_history.append(np.zeros_like(episode['observation'][0][None, :]))
            obs_wrist_history.append(np.zeros_like(episode['observation_wrist'][0][None, :]))
            state_history.append(np.zeros_like(episode['state'][0][None, :]))
            task_embedding_history.append(np.zeros_like(episode['task_embedding'][None, :]))
        obs_agent_history = np.vstack(obs_agent_history)
        obs_wrist_history = np.vstack(obs_wrist_history)
        state_history = np.vstack(state_history)
        task_embedding_history = np.vstack(task_embedding_history)                                
        obs_history = (obs_agent_history, obs_wrist_history, state_history, task_embedding_history)
                                 
        if self.vocab_size is not None:
            tok = episode['token'][idx]
            return (task_embedding, obs_agent, obs_wrist, state, action, tok, action_seq, next_obs_lst, obs_history, pad_idx, pad_mask)
        else:
            return (task_embedding, obs_agent, obs_wrist, state, action, action_seq, next_obs_lst, obs_history, pad_idx, pad_mask)

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader_dist(replay_dir, max_traj_per_task, max_size, batch_size, num_workers,
                       save_snapshot, nstep, nstep_history, discount, rank, world_size,
                        n_code=None, vocab_size=None, min_frequency=None,
                        max_token_length=None):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_traj_per_task,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            nstep_history,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot,
                            rank=rank,
                            world_size=world_size,
                            n_code=n_code,
                            vocab_size=vocab_size,
                            min_frequency=min_frequency,
                            max_token_length=max_token_length)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=False,
                                         worker_init_fn=_worker_init_fn)
    return loader
