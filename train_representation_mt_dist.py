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

import hydra
import numpy as np
import torch
import torch.nn as nn
import mw
import utils
from logger_offline import Logger
from replay_buffer import make_replay_loader_dist
from video import TrainVideoRecorder, VideoRecorder
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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
        
        # TODO: make these parameters, not hard-coded values.
        a_dim = 4
        obs_shape = (3*self.cfg.frame_stack,84,84)

        # NOTE: in this file, the cfg that is being used is offline_mt_representation_config.yaml.
        self.agent = make_agent(obs_shape,
                                a_dim,
                                rank,
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        #print(self.device)
        #### Don't need to load the data in the second stage (calculating BPE)
        if cfg.stage ==2:
            return 
        self.setup()

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, offline=True)
        # create envs
        print('Rank:{} World Size:{}'.format(self.rank, self.world_size))

        offline_data_dirs, counter, self.task_map = [], 0, [[] for i in range(self.world_size)]
        if len(self.cfg.task_names) > 1:
            for task_name in self.cfg.task_names:
    
                #offline_data_dir = '{}/{}_expert500'.format(self.cfg.data_storage_dir, task_name) 
                offline_data_dirs.append(Path(offline_data_dir))
        
        else:
            task_name = self.cfg.task_names[0]
            self.eval_env = mw.make(task_name, self.cfg.frame_stack,
                                    self.cfg.action_repeat, self.cfg.seed, train=False)
            offline_data_dir = self.cfg.offline_data_dir
            
        
        if self.cfg.offline_data_dir == 'none':
            if self.cfg.stage <= 2:
                self.replay_loader = make_replay_loader_dist(
                    offline_data_dirs, self.cfg.replay_buffer_size,
                    self.cfg.batch_size//self.world_size, self.cfg.replay_buffer_num_workers,
                    True, self.cfg.nstep, self.cfg.discount, self.rank, self.world_size)
            else:
                ### Loading tokens after stage 2
                 self.replay_loader = make_replay_loader_dist(
                    offline_data_dirs, self.cfg.replay_buffer_size,
                    self.cfg.batch_size//self.world_size, self.cfg.replay_buffer_num_workers,
                    True, self.cfg.nstep, self.cfg.discount, self.rank, self.world_size,
                    n_code=self.cfg.n_code, vocab_size=self.cfg.vocab_size,
                    min_frequency=self.cfg.min_frequency, max_token_length=self.cfg.max_token_length)
        else:
            #print('Create Data Loader')
            self.replay_loader = make_replay_loader_dist(
                    [Path(self.cfg.offline_data_dir)], self.cfg.replay_buffer_size,
                    self.cfg.batch_size//self.world_size, self.cfg.replay_buffer_num_workers,
                    True, self.cfg.nstep, self.cfg.discount, self.rank, self.world_size,
                    n_code=self.cfg.n_code, vocab_size=self.cfg.vocab_size,
                    min_frequency=self.cfg.min_frequency, max_token_length=self.cfg.max_token_length)

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
        
        if len(code_buffer) == 0:
            ### query the meta/option policy
            meta_action = self.agent.TACO.module.meta_policy(z).max(-1)[1] 
            tok = self.idx_to_tok[int(meta_action.item())]
            try:
                code_buffer = self.tokenizer.decode([tok], verbose=False)
            except:
                print(meta_action)
                assert False
            #codes = self.vocab[meta_action].split('C')[1:]
            #code_buffer = [int(i) for i in codes]
            #code_buffer = [meta_action]    
        
        code_selected = code_buffer.pop(0)
        learned_code  = self.agent.TACO.module.a_quantizer.embedding.weight
        u = learned_code[code_selected, :]
        action = self.agent.TACO.module.decoder(z + u)
        return code_buffer, action.detach().cpu().numpy()[0]

    def eval_mt45(self):
        performance = {}
        task_lst = ['assembly', 'basketball', 'button-press-topdown', 'button-press-topdown-wall', 'button-press', 'button-press-wall', 'coffee-button', 'coffee-pull', 'coffee-push', 'dial-turn', 'disassemble', 'door-close', 'door-open', 'drawer-close', 'drawer-open', 'faucet-open', 'faucet-close', 'hammer', 'handle-press-side', 'handle-press', 'handle-pull-side', 'handle-pull', 'lever-pull', 'peg-insert-side', 'pick-place-wall', 'pick-out-of-hole', 'reach', 'push-back', 'push', 'pick-place', 'plate-slide', 'plate-slide-side', 'plate-slide-back', 'plate-slide-back-side', 'peg-unplug-side', 'soccer', 'stick-push', 'stick-pull', 'push-wall', 'reach-wall', 'shelf-place', 'sweep-into', 'sweep', 'window-open', 'window-close']
        
        for task_name in task_lst:
            env = mw.make(task_name, 3, 2, self.cfg.seed, device_id=self.device, train=False)
            reward_total, success = [], 0
            for i in range(self.cfg.num_eval_episodes):
                time_step = env.reset()
                code_buffer = []
                done, step = False, 0
                #frames = []
                reward = 0
                while not done and step < 200:
                    code_buffer, action = self.act(time_step['observation'], code_buffer)
                    action = np.clip(action, -1, 1.)
                    time_step = env.step(action)
                    obs = time_step['observation']
                    if time_step['success']==1.0:
                        success += 1
                        break
                    step += 1
                    reward += time_step['reward']
                reward_total.append(reward)
            performance[task_name] = success / self.cfg.num_eval_episodes * 100
            if self.rank == 0:
                print("===============Task:{} Number of Eval Episodes:{} Success Rate:{}%===============".format(task_name, self.cfg.num_eval_episodes, success / self.cfg.num_eval_episodes * 100))
        
        if self.rank == 0:
            eval_dir = self.work_dir / 'eval'
            eval_dir.mkdir(exist_ok=True)
            save_dir = eval_dir / '{}.pkl'.format(self.cfg.exp_bc_name)
            with open(save_dir, 'wb') as f:
                pickle.dump(performance, f)
    
    def eval_st(self):
        #print('=====================Begin Evaluation=====================')
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        
        eval_env, task_name = self.eval_env, self.cfg.task_names[0]
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
        eval_dir = self.work_dir / 'eval'
        eval_dir.mkdir(exist_ok=True)
        if self.rank == 0:    
            with open(eval_dir / '{}.pkl'.format(self.cfg.exp_bc_name), 'wb') as f:
                pickle.dump(self.performance, f)
            #print('=======================End Evaluation=======================')
    
    def train_tokenizer(self):
        metrics = None
        while self.global_step < self.cfg.num_train_steps:
            if self.global_step%100 == 0 and self.rank == 0:
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
                # if self.global_step%10000 == 0 and self.rank == 0:
                #     self.save_model()

            self._global_step += 1
            metrics = self.agent.update(self.replay_iter, self.global_step)
            self.logger.log_metrics(metrics, self.global_step, ty='train')

    
    def train_bpe(self):
        self.agent.TACO.train(False)
        task_list = ['assembly', 'basketball', 'button-press-topdown', 'button-press-topdown-wall', 'button-press', 'button-press-wall', 'coffee-button', 'coffee-pull', 'coffee-push', 'dial-turn', 'disassemble', 'door-close', 'door-open', 'drawer-close', 'drawer-open', 'faucet-open', 'faucet-close', 'hammer', 'handle-press-side', 'handle-press', 'handle-pull-side', 'handle-pull', 'lever-pull', 'peg-insert-side', 'pick-place-wall', 'pick-out-of-hole', 'reach', 'push-back', 'push', 'pick-place', 'plate-slide', 'plate-slide-side', 'plate-slide-back', 'plate-slide-back-side', 'peg-unplug-side', 'soccer', 'stick-push', 'stick-pull', 'push-wall', 'reach-wall', 'shelf-place', 'sweep-into', 'sweep', 'window-open', 'window-close']
        lst_traj = []
        for task in task_list:
            path = Path("{}/{}_expert500".format(self.cfg.data_storage_dir, task))
            lst_traj.extend(list(sorted(path.glob('*.npz'))))
        
        
        print('Load {} Trajectories'.format(len(lst_traj)))
        traj_names = []
        ### Train BPE tokenizer
        counter = 0
        corpus = []
        for f in lst_traj:
            counter += 1
            try:
                episode = np.load(f)
            except:
                continue
            obs, action = episode['observation'], episode['action']
            obs = torch.from_numpy(obs).to(self.device)
            action = torch.from_numpy(action).to(self.device)
            z = self.agent.encoder(obs.float())
            u = self.agent.TACO.module.action_encoder(z, action)
            _, _, _, _, min_encoding_indices = self.agent.TACO.module.a_quantizer(u)
            min_encoding_indices = list(min_encoding_indices.reshape(-1).detach().cpu().numpy())
            min_encoding_indices = [int(idx) for idx in min_encoding_indices]
            corpus.append(min_encoding_indices)
            traj_names.append(str(f))
        print('=========Offline Data Tokenized!==========')

        tokenizer = Tokenizer(algo='bpe', vocab_size=self.cfg.vocab_size)
        tokenizer.train(corpus, min_frequency=self.cfg.min_frequency, max_token_length=self.cfg.max_token_length, verbose=True)

        vocab_dir = self.work_dir / 'vocab'
        vocab_dir.mkdir(exist_ok=True)
        with open(vocab_dir / 'vocab_mt45_code{}_vocab{}_minfreq{}_maxtoken{}.pkl'.format(self.cfg.n_code, self.cfg.vocab_size, self.cfg.min_frequency, self.cfg.max_token_length), 'wb') as f:
            pickle.dump([tokenizer, corpus, traj_names], f)

        #### Tokenize Trajectories from 5 Unseen Takss
        lst_traj = []
        task_list = ['box-close', 'hand-insert', 'bin-picking', 'door-lock', 'door-unlock']
        for task in task_list:
            for seed in range(4):
                path = Path("{}/{}_expert3_{}".format(self.cfg.data_storage_dir, task, seed+1))
                lst_traj.extend(list(sorted(path.glob('*.npz'))))

        task_list = ['box-close', 'hand-insert', 'bin-picking', 'door-lock', 'door-unlock']
        for task in task_list:
            for seed in range(4):
                path = Path("{}/{}_expert5_2_{}".format(self.cfg.data_storage_dir, task, seed+1))
                lst_traj.extend(list(sorted(path.glob('*.npz'))))

        task_list = ['box-close', 'hand-insert', 'bin-picking', 'door-lock', 'door-unlock']
        for task in task_list:
            for seed in range(4):
                path = Path("{}/{}_expert10_{}".format(self.cfg.data_storage_dir, task, seed+1))
                lst_traj.extend(list(sorted(path.glob('*.npz'))))
        
        ### Rewrite the trajectory with BPE generated vocabulary
        for f in lst_traj:
            try:
                with np.load(f) as e:
                    episode = dict(e)
            except:
                #print(f)
                continue
            with torch.no_grad():
                obs, action = episode['observation'], episode['action']
                obs = torch.from_numpy(obs).to(self.device)
                action = torch.from_numpy(action).to(self.device)
                z = self.agent.encoder(obs.float())
                u = self.agent.TACO.module.action_encoder(z, action)
                _, _, _, _, min_encoding_indices = self.agent.TACO.module.a_quantizer(u)
                min_encoding_indices = list(min_encoding_indices.reshape(-1).detach().cpu().numpy())
                min_encoding_indices = [int(idx) for idx in min_encoding_indices]
    
            #traj_tok = utils.tokenize_vocab(min_encoding_indices, vocab_lookup, merges)
            traj_tok = [tokenizer.encode(min_encoding_indices[t:], verbose=False)[0] for t in range(obs.shape[0])]
            traj_tok =  np.array(traj_tok, dtype=np.int64).reshape(len(traj_tok), -1)
            episode['code{}_vocab{}_minfreq{}_maxtoken{}'.format(self.cfg.n_code, self.cfg.vocab_size, 
                                                                 self.cfg.min_frequency, self.cfg.max_token_length)] = traj_tok
            utils.save_episode(episode, f)

    
    def train_metapolicy(self):
        metrics = None
        with open(self.work_dir / 'vocab' / 'vocab_mt45_code{}_vocab{}_minfreq{}_maxtoken{}.pkl'.format(self.cfg.n_code, self.cfg.vocab_size, self.cfg.min_frequency, self.cfg.max_token_length), 'rb') as f:
            loaded_data = pickle.load(f)
            self.tokenizer, corpus, traj_names = loaded_data

        #### Tokenizer the given trajectories
        lst_traj = []
        path = Path(self.cfg.offline_data_dir)
        lst_traj= list(sorted(path.glob('*.npz')))
        self.tok_to_idx = dict()
        self.idx_to_tok = []
        for f in lst_traj:
            with np.load(f) as e:
                episode = dict(e)
            with torch.no_grad():
                obs, action = episode['observation'], episode['action']
                obs = torch.from_numpy(obs).to(self.device)
                action = torch.from_numpy(action).to(self.device)
                z = self.agent.encoder(obs.float())
                u = self.agent.TACO.module.action_encoder(z, action)
                _, _, _, _, min_encoding_indices = self.agent.TACO.module.a_quantizer(u)
                min_encoding_indices = list(min_encoding_indices.reshape(-1).detach().cpu().numpy())
                min_encoding_indices = [int(idx) for idx in min_encoding_indices]
            
            traj_tok = [self.tokenizer.encode(min_encoding_indices[t:], verbose=False)[0] for t in range(obs.shape[0])]
            for tok in traj_tok:
                if not tok in self.tok_to_idx:
                    self.tok_to_idx[tok] = len(self.tok_to_idx)
                    self.idx_to_tok.append(tok)
            
        
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
        tok_to_code = lambda tok: self.tokenizer.decode([int(tok.item())], verbose=False)[0]
        tok_to_idx  = lambda tok: self.tok_to_idx[int(tok.item())]
        
        #index_fn = lambda x: int(self.vocab[x.item()].split('C')[1])
        while self.global_step < self.cfg.num_train_steps:
            if self.global_step%1000 == 0 and self.rank == 0:
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    print('DECODER_LOSS:{}, META_POLICY_LOSS:{}'.format(metrics['decoder_loss'], metrics['meta_policy_loss'])) 
                    elapsed_time, total_time = self.timer.reset()

                # reset env
                # try to save snapshot
                if self.cfg.save_snapshot and self.rank == 0:
                    self.save_snapshot(self.cfg.stage)

            self._global_step += 1
            metrics = self.agent.update_metapolicy(self.replay_iter, self.global_step, tok_to_code, tok_to_idx)
        
            if self.global_step%self.cfg.eval_freq == 0:
                if len(self.cfg.task_names) == 1:
                    self.eval_st()
                else:
                    self.eval_mt45()
        if self.global_step%self.cfg.eval_freq == 0:
            if len(self.cfg.task_names) == 1:
                self.eval_st()
            else:
                
                self.eval_mt45()
    
    # def train_multitask_bc(self):
    #     metrics = None
    #     while self.global_step < self.cfg.num_train_steps:
    #         if self.global_step%100 == 0 and self.rank == 0:
    #             # wait until all the metrics schema is populated
    #             if metrics is not None:
    #                 # log stats
    #                 print('DECODER_LOSS:{}'.format(metrics['decoder_loss']))
    #                 elapsed_time, total_time = self.timer.reset()
    #                 with self.logger.log_and_dump_ctx(self.global_step,
    #                                                   ty='train') as log:
    #                     log('total_time', total_time)
    #                     log('step', self.global_step)

    #             if self.global_step > 0:
    #                 if self.cfg.save_snapshot and self.rank == 0:
    #                     self.save_snapshot(1)

    #         self._global_step += 1
    #         metrics = self.agent.update_multitask_bc(self.replay_iter, self.global_step)
    #         self.logger.log_metrics(metrics, self.global_step, ty='train')

    
    def save_snapshot(self, stage):
        if stage == 1:
            snapshot = self.work_dir / 'snapshot.pt'
        else:
            snapshot = self.work_dir / 'snapshot_vocab{}.pt'.format(self.cfg.vocab_size)
                                                                   
        keys_to_save = ['agent', '_global_step']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        #payload = self.agent.TACO.state_dict()
        with snapshot.open('wb') as f:
            torch.save(payload, f)
            
    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        #try:
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        self.__dict__['agent'] = payload['agent']
        if self.cfg.stage == 1:
            self.__dict__['_global_step'] = payload['_global_step']
        #state_dict = torch.load(self.work_dir / 'model_10.pt', map_location=torch.device('cuda:{}'.format(self.rank)))
        self.agent.device = self.device
        self.agent.TACO.device = self.device
        self.agent.TACO.to(self.device)
        print('Resuming Snapshopt')
        
        # ### Reinitialize the meta-policy head
        # if self.cfg.reinit_metapolicy:
        #     self.agent.encoder.eval()
        #     meta_policy = nn.Sequential(
        #     nn.Linear(self.cfg.feature_dim, self.cfg.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.cfg.hidden_dim, self.cfg.vocab_size)
        #     ).to(self.device)
        #     meta_policy.train(True)
        #     meta_policy.apply(utils.weight_init)
        #     self.agent.TACO.module.meta_policy = meta_policy
        #     self.agent.taco_opt = torch.optim.Adam(self.agent.TACO.parameters(), lr=self.cfg.lr)
        # self.agent.taco_opt = torch.optim.Adam(self.agent.TACO.parameters(), lr=self.cfg.lr)
        
    
    def save_encoder(self):
        counter = (self.global_step // 10000) + 1
        snapshot = self.work_dir / "encoder_{}.pt".format(counter)
        
        payload = self.__dict__['agent'].encoder.state_dict()
        with snapshot.open('wb') as f:
            torch.save(payload, f)


# RANK and WORLD_SIZE are needed to initialize the distributed data parallel training setup.
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
    if snapshot.exists() and cfg.stage > 1:
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    if cfg.train_multitask_bc:
        workspace.train_multitask_bc()
    else:
        # The training pipeline consists of 3 stages.
        if cfg.stage == 1:
            workspace.train_tokenizer()
        elif cfg.stage == 2:
            workspace.train_bpe()
        elif cfg.stage == 3:
            workspace.train_metapolicy()
        else:
            raise Error
    destroy_process_group()

def wrapper(rank, world_size):
    global RANK, WORLD_SIZE
    RANK = rank
    WORLD_SIZE = world_size
    main()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(wrapper, args=(world_size,), nprocs=world_size)
