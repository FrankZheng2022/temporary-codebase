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
from dm_env import specs

#import dmc
import mw
from metaworld.policies import *
import utils
import shutil
from logger import Logger
from collect_data_replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

def clean(path):
    try:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    except:
        return



def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        # self.agent = make_agent(self.train_env.observation_spec(),
        #                         self.train_env.action_spec(),
        #                         self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create envs
        if self.cfg.domain == 'dmc':
            self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                      self.cfg.action_repeat, self.cfg.seed)
        elif self.cfg.domain == 'metaworld':
            self.train_env = mw.make(self.cfg.task_name, self.cfg.frame_stack,
                                      self.cfg.action_repeat, self.cfg.seed,
                                     train=True, device_id=-1, cam_name=self.cfg.cam_name)
            
        else:
            assert False
       
        clean(self.cfg.offline_data_dir)
        offline_data_dir = Path(self.cfg.offline_data_dir)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.state_spec(), #for dmc
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  offline_data_dir / self.cfg.task_name,
                                                  store_only_success=True)

        self.replay_loader = make_replay_loader(
            offline_data_dir, self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            True, self.cfg.nstep, self.cfg.multistep, self.cfg.discount)
        self._replay_iter = None

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def generate(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        #print('Reset Environment')
        self.replay_storage.add(time_step)
        metrics = None
        self._global_episode = 0
        success = False
        while self._global_episode < self.cfg.num_expert_trajectories:
            if time_step['success'] == 1.0:
                success = True
            if time_step.last() or success:
                if success:
                    self._global_episode += 1
                    if self._global_episode % 5 == 0:
                        print("Generate {} Trajectories".format(self._global_episode))

                # reset env
                success = False
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                episode_step = 0
                episode_reward = 0

            action = self.agent.get_action(self.train_env.state)
            action = np.random.normal(action, self.cfg.noise * 2.)
            action = np.clip(action, -1., 1.)
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            episode_step += 1
            self._global_step += 1


    def load_snapshot(self, model_dir):
        snapshot = Path(model_dir) / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
    
    def load_mw_policy(self, task_name):
        if task_name == 'peg-insert-side':
            self.agent = SawyerPegInsertionSideV2Policy()
        else:
            task_name = task_name.split('-')
            task_name = [s.capitalize() for s in task_name]
            task_name = "Sawyer" + "".join(task_name) + "V2Policy"
            self.agent = eval(task_name)()

@hydra.main(config_path='cfgs', config_name='generate_expert_config')
def main(cfg):
    from generate_expert_dataset import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    #if cfg.domain == 'dmc':
    #workspace.load_snapshot(cfg.model_dir)
    # elif cfg.domain == 'metaworld':
    workspace.load_mw_policy(cfg.task_name)
    # else:
    #     assert False
    workspace.generate()


if __name__ == '__main__':
    main()
 






