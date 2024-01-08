# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import distutils.dir_util
import hydra
import numpy as np
import time
import torch
import torch.nn as nn
import random
import libero_wrapper
from libero.libero import benchmark
import utils
from logger_offline import Logger
from replay_buffer import make_replay_loader_dist
from video import TrainVideoRecorder, VideoRecorder
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group, gather
#from bpe import compute_pair_freqs, merge_pair, tokenize
from collections import defaultdict
import copy
import pickle
import io
import torch.nn.functional as F
from tokenizer_api import Tokenizer
import time

torch.backends.cudnn.benchmark = True

def ddp_setup(rank, world_size, port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "{}".format(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def make_agent(obs_shape, action_dim, rank, cfg):
    cfg.obs_shape = obs_shape
    cfg.action_dim = action_dim
    device_ids = list(range(torch.cuda.device_count()))
    cfg.device = device_ids[rank]
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg, rank, world_size):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.rank = rank
        self.world_size = world_size
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)

        device_ids = list(range(torch.cuda.device_count()))
        self.device = device_ids[rank]
        self.results_dir = Path(self.cfg.results_dir)

        a_dim = self.cfg.action_dim
        obs_shape = [3*self.cfg.frame_stack]+list(self.cfg.img_res)  #(3*self.cfg.frame_stack,84,84)
        self.agent = make_agent(obs_shape,
                                a_dim,
                                rank,
                                self.cfg.agent)
        
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.cfg.downstream_task_suite]()
        task = task_suite.get_task(int(self.cfg.downstream_task_name))
        self.eval_env = libero_wrapper.make(self.cfg.downstream_task_name, 
                                            self.cfg.downstream_task_suite, seed=self.cfg.seed, 
                                            frame_stack=self.cfg.frame_stack)
        self.eval_env.task_name = task.name
        self.eval_env.task_embedding = libero_wrapper.get_task_embedding(task.language)

    def act(self, env, obs, code_buffer):
        
        obs_agent = obs.agentview
        obs_wrist = obs.wristview
        state     = obs.state 
        task_embedding = env.task_embedding
        
        
        task_embedding = torch.torch.as_tensor(task_embedding, device=self.device)
        obs_agent = torch.torch.as_tensor(obs_agent, device=self.device).unsqueeze(0)
        obs_wrist = torch.torch.as_tensor(obs_wrist, device=self.device).unsqueeze(0)
        state     = torch.torch.as_tensor(state, device=self.device).unsqueeze(0)
        
        # z_agent = self.agent.encoders[0](obs_agent.float(), langs=task_embedding)
        # z_wrist = self.agent.encoders[1](obs_wrist.float(), langs=task_embedding)
        # state   = self.agent.encoders[2](state.float())
        # z = torch.concatenate([z_agent, z_wrist, state, task_embedding], dim=-1)
        z = self.agent.encode_obs(obs_agent, obs_wrist, state, task_embedding)
        
        ### For Vanilla BC, use decoder to directly predict the raw action
        if self.cfg.bc:
            if self.cfg.decoder_type == 'deterministic':
                action = self.agent.TACO.module.decoder(z)
            elif self.cfg.decoder_type == 'gmm':
                action = self.agent.TACO.module.decoder(z).sample()
            else:
                raise Exception
            return [], action.detach().cpu().numpy()[0]
        if self.cfg.non_bpe:
            action_code = self.agent.TACO.module.meta_policy(z).max(-1)[1]
            learned_code  = self.agent.TACO.module.a_quantizer.embedding.weight
            u = learned_code[action_code, :]
            action = self.agent.TACO.module.decoder(z + u)
            return [], action.detach().cpu().numpy()[0]
        
        if len(code_buffer) == 0:
            ### query the meta/option policy
            meta_action = self.agent.TACO.module.meta_policy(z).max(-1)[1]
            tok = self.idx_to_tok[int(meta_action.item())]
            try:
                code_buffer = self.tokenizer.decode([tok], verbose=False)
            except:
                print('Error occured when choosing meta action:{}'.format(meta_action))
                assert False

        code_selected = code_buffer.pop(0)
        learned_code  = self.agent.TACO.module.a_quantizer.embedding.weight
        u = learned_code[code_selected, :]
        action = self.agent.TACO.module.decoder(z + u)
        return code_buffer, action.detach().cpu().numpy()[0]

    def evaluate_st(self):
        eval_start_time = time.time()
        print('=====================Begin Evaluation=====================')
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        eval_env, task_name = self.eval_env, self.cfg.downstream_task_name
        counter, episode, success = 0, 0, 0
        while eval_until_episode(episode):
            time_step = eval_env.reset()
            step, code_buffer = 0, []
            while step < self.cfg.eval_max_steps:
                if time_step['done']:
                    success += 1
                    break
                with torch.no_grad():
                    code_buffer, action = self.act(eval_env, time_step, code_buffer)
                time_step = eval_env.step(action)
                step += 1
            episode += 1
        
        print(f'Evaluation Time:{time.time()-eval_start_time}s Success Rate:{success/self.cfg.num_eval_episodes*100}%')
        print('=====================End Evaluation=====================')
        
        
    def evaluate_mt(self):
        eval_start_time = time.time()
        print('=====================Begin Evaluation=====================')
        
        for i in range(int(self.cfg.downstream_task_name)*10, int(self.cfg.downstream_task_name+1)*10):
            
            eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
            
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[self.cfg.downstream_task_suite]()
            task = task_suite.get_task(i)
            eval_env = libero_wrapper.make(i, self.cfg.downstream_task_suite, 
                                           seed=self.cfg.seed, frame_stack=self.cfg.frame_stack)
            eval_env.task_name = task.name
            eval_env.task_embedding = libero_wrapper.get_task_embedding(task.language)
            task_name = task.name
            counter, episode, success = 0, 0, 0
            while eval_until_episode(episode):
                time_step = eval_env.reset()
                step, code_buffer = 0, []
                while step < self.cfg.eval_max_steps:
                    if time_step['done']:
                        success += 1
                        break
                    with torch.no_grad():
                        code_buffer, action = self.act(eval_env, time_step, code_buffer)
                    time_step = eval_env.step(action)
                    step += 1
                episode += 1

            print(f'Task:{task_name} Evaluation Time:{time.time()-eval_start_time}s Success Rate:{success/self.cfg.num_eval_episodes*100}%', flush=True)
        print('=====================End Evaluation=====================')
        # if self.rank == 0:
        #     with open(self.eval_dir / '{}.pkl'.format(self.cfg.exp_bc_name), 'wb') as f:
        #         pickle.dump(self.performance, f)
            #print('=======================End Evaluation=======================')

    
    def load_snapshot(self):
        snapshot = self.results_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        self.__dict__['agent'] = payload['agent']
        self.agent.device = self.device
        self.agent.TACO.device = self.device
        self.agent.TACO.to(self.device)
        print('Resuming Snapshopt')



RANK = None
WORLD_SIZE = None

@hydra.main(config_path='cfgs', config_name='offline_mt_representation_config')
def main(cfg):
    global RANK, WORLD_SIZE
    ddp_setup(RANK, WORLD_SIZE, cfg.port)
    from evaluate import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg, RANK, WORLD_SIZE)
    root_dir = Path.cwd()
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.evaluate_st()
    destroy_process_group()

def wrapper(rank, world_size, cfg):
    global RANK, WORLD_SIZE
    RANK = rank
    WORLD_SIZE = world_size
    print(f'WORLD SIZE: {world_size}, RANK: {rank}')
    main(cfg)

def main_mp_launch_helper(cfg=None):
    world_size = torch.cuda.device_count()
    if world_size==1:  # single GPU
        wrapper(0, 1, cfg)  # don't use multiprocessing
    else:
        mp.spawn(wrapper, args=(world_size, cfg), nprocs=world_size)

if __name__ == '__main__':
    main_mp_launch_helper()
