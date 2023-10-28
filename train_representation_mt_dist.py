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
import mw
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


def construct_task_data_path(root_dir, task_name, task_data_dir_suffix):
    return Path(root_dir) / (task_name+('' if not task_data_dir_suffix or task_data_dir_suffix == 'None' else task_data_dir_suffix))


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

        a_dim = self.cfg.action_dim
        obs_shape = [3*self.cfg.frame_stack]+list(self.cfg.img_res)  #(3*self.cfg.frame_stack,84,84)
        self.agent = make_agent(obs_shape,
                                a_dim,
                                rank,
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.results_dir = Path(self.cfg.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.pretraining_data_dirs = []

        self.pretraining_data_dirs = []
        if self.cfg.stage < 3:
            #for task_name in self.cfg.task_names:
            for task_name in utils.task_names(self.cfg.split):
                offline_data_dir = construct_task_data_path(self.cfg.data_storage_dir, task_name, self.cfg.task_data_dir_suffix)
                self.pretraining_data_dirs.append(offline_data_dir)
        else:
            task_name = self.cfg.downstream_task_name
            self.eval_env = mw.make(task_name, self.cfg.frame_stack,
                                    self.cfg.action_repeat, self.cfg.seed, train=False)

        #### Don't need to load the data in the second stage (calculating BPE)
        assert self.cfg.stage in [1, 2, 3], "Stage must be 1, 2, or 3."

        if self.cfg.stage < 3:
            self.eval_dir = self.results_dir
        else:
            self.eval_dir = self.results_dir / 'eval' / f'stage_{self.cfg.stage}' / self.cfg.downstream_task_name
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.stage == 2:
            return
        self.setup_replay_buffer()


    def setup_replay_buffer(self):
        # create logger
        log_dir = self.work_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(log_dir, use_tb=self.cfg.use_tb, offline=True)
        # create envs
        print('Rank:{} World Size:{}'.format(self.rank, self.world_size))

        if self.cfg.stage == 1:
            self.replay_loader = make_replay_loader_dist(
                self.pretraining_data_dirs, self.cfg.max_traj_per_task, self.cfg.replay_buffer_size,
                self.cfg.batch_size//self.world_size, self.cfg.replay_buffer_num_workers,
                True, self.cfg.nstep, self.cfg.discount, self.rank, self.world_size)
        elif self.cfg.stage == 3:
            downstream_data_path = construct_task_data_path(self.cfg.data_storage_dir, self.cfg.downstream_task_name, self.cfg.task_data_dir_suffix)
            print(f"Loading target task data from {downstream_data_path}")
            if self.cfg.bc:
                self.replay_loader = make_replay_loader_dist(
                    [downstream_data_path], self.cfg.max_traj_per_task, self.cfg.replay_buffer_size,
                    self.cfg.batch_size//self.world_size, self.cfg.replay_buffer_num_workers,
                    True, self.cfg.nstep, self.cfg.discount, self.rank, self.world_size)
            else:
                self.replay_loader = make_replay_loader_dist(
                    [downstream_data_path], self.cfg.max_traj_per_task, self.cfg.replay_buffer_size,
                    self.cfg.batch_size//self.world_size, self.cfg.replay_buffer_num_workers,
                    True, self.cfg.nstep, self.cfg.discount, self.rank, self.world_size,
                    n_code=self.cfg.n_code, vocab_size=self.cfg.vocab_size,
                    min_frequency=self.cfg.min_frequency, max_token_length=self.cfg.max_token_length)
        else:
            assert self.cfg.stage != 2, "You shouldn't set up the replay buffer for stage 2. Most likely you ended up here due to a logic bug."

        print('Finish Reading Data')
        self._replay_iter = None
        self.performance = []


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter= iter(self.replay_loader)
        return self._replay_iter

    def act(self, obs, code_buffer):
        obs = torch.from_numpy(obs).to(self.device)
        z = self.agent.encoder(obs.unsqueeze(0))
        
        ### For Vanilla BC, use decoder to directly predict the raw action
        if self.cfg.bc:
            action = self.agent.TACO.module.decoder(z)
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

    def eval_st(self):
        #print('=====================Begin Evaluation=====================')
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        eval_env, task_name = self.eval_env, self.cfg.downstream_task_name
        counter, step, episode, total_reward, success = 0, 0, 0, 0, 0
        while eval_until_episode(episode):
            time_step = eval_env.reset()
            code_buffer = []
            while not time_step.last():
                if time_step['success'] == 1.0:
                    success += 1
                    break
                if np.max(time_step['observation']) == 0.:
                    print("Error with Observation, Quit")
                    assert False
                with torch.no_grad():
                    code_buffer, action = self.act(time_step.observation, code_buffer)
                time_step = eval_env.step(action)
                total_reward += time_step.reward
                step += 1
            episode += 1

        print('Success Rate:{}'.format(success/self.cfg.num_eval_episodes*100))
        self.performance.append(success/self.cfg.num_eval_episodes*100)
        if self.rank == 0:
            with open(self.eval_dir / '{}.pkl'.format(self.cfg.exp_bc_name), 'wb') as f:
                pickle.dump(self.performance, f)
            #print('=======================End Evaluation=======================')

    def pretrain_models(self):
        metrics = None
        start_train_block_time = time.time()
        while self.global_step < self.cfg.num_train_steps:
            if self.global_step%self.cfg.eval_freq == 0 and self.rank == 0:
                print(f"\nPretraining for {self.global_step} steps of {self.cfg.batch_size}-sized batches has takes {time.time() - start_train_block_time}s.")
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    print('SPR_LOSS:{}, QUANTIZE_LOSS:{}, DECODER_LOSS:{}, META_POLICY_LOSS:{}'.format(metrics['spr_loss'], metrics['quantize_loss'], metrics['decoder_loss'], metrics['meta_policy_loss']))
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.global_step,
                                                      ty='train') as log:
                        log('total_time', total_time)
                        log('step', self.global_step)

                # reset env
                # try to save snapshot
                if self.cfg.save_snapshot and self.rank == 0:
                    self.save_snapshot(self.cfg.stage)

            self._global_step += 1
            metrics = self.agent.update(self.replay_iter, self.global_step)
            self.logger.log_metrics(metrics, self.global_step, ty='train')

        dest_log_dir = self.results_dir / 'logs'
        distutils.dir_util.copy_tree(str(self.logger._log_dir), str(dest_log_dir))

    
    def train_bpe(self):
        self.agent.n_code = self.cfg.n_code
        self.agent.TACO.train(False)
        lst_traj = []
        for task_dir in self.pretraining_data_dirs:
            lst_traj.extend(utils.choose(list(sorted(task_dir.glob('*.npz'))), self.cfg.max_traj_per_task))

        print('Loaded {} trajectories'.format(len(lst_traj)))
        ### Train BPE tokenizer
        counter = 0
        traj_names, corpus, code_distance_lst = [], [], []
        for f in lst_traj:
            counter += 1
            try:
                episode = np.load(f)
                obs, action = episode['observation'], episode['action']
            except:
                continue
            obs = torch.from_numpy(obs).to(self.device)
            action = torch.from_numpy(action).to(self.device)
            z = self.agent.encoder(obs.float())
            with torch.no_grad():
                code_distance = self.agent.cal_distance(z)
                code_distance_lst.append(code_distance[None, :])
            
            u = self.agent.TACO.module.action_encoder(z, action)
            _, _, _, _, min_encoding_indices = self.agent.TACO.module.a_quantizer(u)
            min_encoding_indices = list(min_encoding_indices.reshape(-1).detach().cpu().numpy())
            min_encoding_indices = [int(idx) for idx in min_encoding_indices]
            ### calculate code_distance
            corpus.append(min_encoding_indices)
            # for t in range(len(min_encoding_indices)):
            #     corpus.append(min_encoding_indices[t:])
            traj_names.append(str(f))

            if counter % 100 == 0:
                print(f"Processed {counter} trajectories")

        print('=========Offline Data Tokenized!==========')
        
        
        ### Train tokenizer on the tokenized pretraining trajectories
        tokenizer = Tokenizer(algo='bpe', vocab_size=self.cfg.vocab_size)
        tokenizer.train(corpus, min_frequency=self.cfg.min_frequency, max_token_length=self.cfg.max_token_length, verbose=True)
        
        ### compute the N by N distance matrix for codes
        code_distance = np.mean(np.vstack(code_distance_lst), axis=0)
        
        ### Now calculate the distance matrix for tokens (subtract 1 due to UNK token)
        ### (Note: this matrix is not symmetric)
        tok_distance = np.zeros((tokenizer.vocab_size-1, tokenizer.vocab_size-1))
        for i in range(1, tokenizer.vocab_size):
            code_seq_i = tokenizer.decode([i], verbose=False)
            for j in range(1, tokenizer.vocab_size):
                code_seq_j = tokenizer.decode([j], verbose=False)
                dist = utils.cal_tok_dist(code_seq_i, code_seq_j, code_distance)
                tok_distance[i-1][j-1] = dist
        
        ### Save pretrained tokenizer
        vocab_dir = self.results_dir / 'vocab'
        vocab_dir.mkdir(parents=True, exist_ok=True)
        with open(vocab_dir / 'vocab_mt45_code{}_vocab{}_minfreq{}_maxtoken{}.pkl'.format(self.cfg.n_code, self.cfg.vocab_size, self.cfg.min_frequency, self.cfg.max_token_length), 'wb') as f:
            pickle.dump([tokenizer, corpus, traj_names, code_distance, tok_distance], f)
    

    def train_metapolicy(self):
        self.agent.n_code = self.cfg.n_code
        self.agent.alpha  = self.cfg.alpha
        metrics = None
        vocab_dir = self.results_dir / 'vocab'
        with open(vocab_dir / 'vocab_mt45_code{}_vocab{}_minfreq{}_maxtoken{}.pkl'.format(self.cfg.n_code, self.cfg.vocab_size, self.cfg.min_frequency, self.cfg.max_token_length), 'rb') as f:
            loaded_data = pickle.load(f)
            self.tokenizer, corpus, traj_names, code_distance, tok_distance = loaded_data
        #### Tokenizer the given trajectories and check the number of unique tokens in the given demonstration trajectories
        print("========= Tokenizing the downstream data... ==========")
        replay_buffer = self.replay_loader.dataset  # HACK
        self.tok_to_idx = dict() ### Token, Index Lookup
        self.idx_to_tok = []
        code_distance_lst = []
        for episode in replay_buffer._episodes.values():
            with torch.no_grad():
                obs, action = episode['observation'], episode['action']
                obs = torch.from_numpy(obs).to(self.device)
                action = torch.from_numpy(action).to(self.device)
                z = self.agent.encoder(obs.float())
                code_distance = self.agent.cal_distance(z)
                code_distance_lst.append(code_distance[None, :])
                u = self.agent.TACO.module.action_encoder(z, action)
                _, _, _, _, min_encoding_indices = self.agent.TACO.module.a_quantizer(u)
                min_encoding_indices = list(min_encoding_indices.reshape(-1).detach().cpu().numpy())
                min_encoding_indices = [int(idx) for idx in min_encoding_indices]
            
            
            traj_tok = [self.tokenizer.encode(min_encoding_indices[t:], verbose=False)[0] for t in range(obs.shape[0])]
            episode['token'] = traj_tok
            for tok in traj_tok:
                if not tok in self.tok_to_idx:
                    self.tok_to_idx[tok] = len(self.tok_to_idx)
                    self.idx_to_tok.append(tok)
        
        code_distance = np.mean(np.vstack(code_distance_lst), axis=0)   
        ### Calculate distance_metric
        idx_distance = torch.zeros(len(self.idx_to_tok), len(self.idx_to_tok))
        for i in range(len(self.idx_to_tok)):
            code_seq_i = self.tokenizer.decode([self.idx_to_tok[i]], verbose=False)
            for j in range(len(self.idx_to_tok)):
                code_seq_j = self.tokenizer.decode([self.idx_to_tok[j]], verbose=False)
                idx_distance[i][j] = utils.cal_tok_dist(code_seq_i, code_seq_j, code_distance)
        idx_distance = idx_distance.to(self.device)
        
        print(f"========= Initiaizing the model... ==========")
        self.agent.train(False)
        meta_policy = nn.Sequential(
            nn.Linear(self.cfg.feature_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, len(self.tok_to_idx))
        ).to(self.device)
        meta_policy.train(True)
        meta_policy.apply(utils.weight_init)
        self.agent.TACO.module.meta_policy = meta_policy
        self.agent.taco_opt = torch.optim.Adam(self.agent.TACO.parameters(), lr=self.cfg.lr)
        tok_to_code = lambda tok: self.tokenizer.decode([int(tok.item())], verbose=False) ### Token =>  First Code
        tok_to_idx  = lambda tok: self.tok_to_idx[int(tok.item())] ### Token => Index

        print(f"========= Finetuning for {self.cfg.num_train_steps} steps... ==========")
        start_train_block_time = time.time()
        while self.global_step < self.cfg.num_train_steps:
            if self.global_step%self.cfg.eval_freq == 0 and self.rank == 0:
                print(f"\nTraining for {self.global_step} steps of {self.cfg.batch_size}-sized batches has takes {time.time() - start_train_block_time}s (including eval time).")
                #print('CE loss:{}'.format(ce_loss))
                #wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    print('DECODER_LOSS:{}, META_POLICY_LOSS:{}'.format(metrics['decoder_loss'], metrics['meta_policy_loss']))
                    elapsed_time, total_time = self.timer.reset()

                # reset env
                # try to save snapshot
                if self.cfg.save_snapshot and self.rank == 0:
                    self.save_snapshot(self.cfg.stage)

            self._global_step += 1
            metrics = self.agent.update_metapolicy(self.replay_iter, self.global_step, tok_to_code, tok_to_idx, self.idx_to_tok, idx_distance, cross_entropy=self.cfg.cross_entropy)

            if self.global_step>5000 and self.global_step%self.cfg.eval_freq == 0:
                # TODO: we need to leave just one eval method, which should be callable on any number of downstream tasks.
                start_eval_block_time = time.time()
                self.eval_st()
                print(f"Evaluation on {self.cfg.num_eval_episodes} episodes took {time.time() - start_eval_block_time}s.")

                
    def train_metapolicy_nonbpe(self):
        metrics = None
        
        ### Tokenize the trajectory
        print("========= Tokenizing the downstream data... ==========")
        replay_buffer = self.replay_loader.dataset
        for episode in replay_buffer._episodes.values():
            with torch.no_grad():
                obs, action = episode['observation'], episode['action']
                obs = torch.from_numpy(obs).to(self.device)
                action = torch.from_numpy(action).to(self.device)
                z = self.agent.encoder(obs.float())
                u = self.agent.TACO.module.action_encoder(z, action)
                _, _, _, _, min_encoding_indices = self.agent.TACO.module.a_quantizer(u)
                min_encoding_indices = list(min_encoding_indices.reshape(-1).detach().cpu().numpy())
                min_encoding_indices = [int(idx) for idx in min_encoding_indices]
            episode['token'] = min_encoding_indices
        
        print(f"========= Finetuning for {self.cfg.num_train_steps} steps... ==========")
        start_train_block_time = time.time()
        while self.global_step < self.cfg.num_train_steps:
            if self.global_step%self.cfg.eval_freq == 0 and self.rank == 0:
                print(f"\nTraining for {self.global_step} steps of {self.cfg.batch_size}-sized batches has takes {time.time() - start_train_block_time}s (including eval time).")
                if metrics is not None:
                    # log stats
                    print('DECODER_LOSS:{}, META_POLICY_LOSS:{}'.format(metrics['decoder_loss'], metrics['meta_policy_loss']))
                    elapsed_time, total_time = self.timer.reset()

                # reset env
                # try to save snapshot
                if self.cfg.save_snapshot and self.rank == 0:
                    self.save_snapshot(self.cfg.stage)

            self._global_step += 1
            metrics = self.agent.update_metapolicy_nonbpe(self.replay_iter, self.global_step)

            if self.global_step>5000 and self.global_step%self.cfg.eval_freq == 0:
                # TODO: we need to leave just one eval method, which should be callable on any number of downstream tasks.
                start_eval_block_time = time.time()
                self.eval_st()
                print(f"Evaluation on {self.cfg.num_eval_episodes} episodes took {time.time() - start_eval_block_time}s.")

            
    def train_bc(self):
        metrics = None
        start_train_block_time = time.time()
        while self.global_step < self.cfg.num_train_steps:
            if self.global_step%self.cfg.eval_freq == 0 and self.rank == 0:
                print(f"\nTraining for {self.global_step} steps of {self.cfg.batch_size}-sized batches has takes {time.time() - start_train_block_time}s (including eval time).")
                if metrics is not None:
                    # log stats
                    print('BC_LOSS:{}'.format(metrics['bc_loss']))
                    elapsed_time, total_time = self.timer.reset()

                # reset env
                # try to save snapshot
                if self.cfg.save_snapshot and self.rank == 0:
                    self.save_snapshot(self.cfg.stage)
            
            if self.global_step>5000 and self.global_step%self.cfg.eval_freq == 0:
                start_eval_block_time = time.time()
                self.eval_st()
                print(f"Evaluation on {self.cfg.num_eval_episodes} episodes took {time.time() - start_eval_block_time}s.")
                
            self._global_step += 1
            metrics = self.agent.update_bc(self.replay_iter, self.global_step)
            self.logger.log_metrics(metrics, self.global_step, ty='train')

        dest_log_dir = self.results_dir / 'logs'
        distutils.dir_util.copy_tree(str(self.logger._log_dir), str(dest_log_dir))
            
    def save_snapshot(self, stage):
        if stage == 1:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            snapshot = self.results_dir / 'snapshot.pt'
        elif stage == 2:
            snapshot = self.results_dir / 'snapshot_vocab{}.pt'.format(self.cfg.vocab_size)
        else:
            snapshot = self.eval_dir / 'snapshot_vocab{}_{}.pt'.format(self.cfg.vocab_size, self.cfg.seed)

        keys_to_save = ['agent', '_global_step']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        #payload = self.agent.TACO.state_dict()
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.results_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        self.__dict__['agent'] = payload['agent']
        if self.cfg.stage == 1:
            self.__dict__['_global_step'] = payload['_global_step']
        self.agent.device = self.device
        self.agent.TACO.device = self.device
        self.agent.TACO.to(self.device)
        print('Resuming Snapshopt')


    def save_encoder(self):
        counter = (self.global_step // 10000) + 1
        snapshot = self.results_dir / "encoder_{}.pt".format(counter)

        payload = self.__dict__['agent'].encoder.state_dict()
        with snapshot.open('wb') as f:
            torch.save(payload, f)

RANK = None
WORLD_SIZE = None

@hydra.main(config_path='cfgs', config_name='offline_mt_representation_config')
def main(cfg):
    global RANK, WORLD_SIZE
    ddp_setup(RANK, WORLD_SIZE, cfg.port)
    from train_representation_mt_dist import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg, RANK, WORLD_SIZE)
    root_dir = Path.cwd()
    snapshot = root_dir / 'snapshot.pt'
    if cfg.load_snapshot:
        if snapshot.exists() and cfg.stage > 1:
            print(f'resuming: {snapshot}')
            workspace.load_snapshot()
    ### Train Vanilla BC if bc=True
    if cfg.bc:
        workspace.train_bc()
    if cfg.stage == 1:
        workspace.pretrain_models()
    elif cfg.stage == 2:
        workspace.train_bpe()
    elif cfg.stage == 3:
        if cfg.non_bpe:
            workspace.train_metapolicy_nonbpe()
        else:
            workspace.train_metapolicy()
    else:
        raise ValueError(f"Invalid stage: {cfg.stage}")
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
