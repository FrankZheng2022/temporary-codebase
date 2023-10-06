import hydra
import time
from pathlib import Path
from train_representation_mt_dist import main_mp_launch_helper as train_main


@hydra.main(config_path='cfgs', config_name='full_pipeline_config')
def main(cfg):
    timenow = time.strftime("%H.%M.%S--%m-%d-%Y")

    stage_shared_overrides = [
                                f"feature_dim={cfg.feature_dim}",
                                f"data_storage_dir={cfg.data_storage_dir}",
                                f"task_data_dir_suffix={cfg.task_data_dir_suffix}",
                                f"results_dir={Path(cfg.results_dir) / timenow}",
                                f"n_code={cfg.n_code}",
                                f"vocab_size={cfg.vocab_size}",
                                f"hidden_dim={cfg.hidden_dim}",
                                f"nstep={cfg.nstep}",
                                f"exp_name={cfg.exp_name}",
                                f"obs_dependent={cfg.obs_dependent}",
                                f"min_frequency={cfg.min_frequency}",
                                f"max_token_length={cfg.max_token_length}",
                                f"seed={cfg.seed}",
                             ]

    stage_1_overrides = [
                            #--- Stage-specific *user-specified* overrides ---
                            f"stage=1",
                            f"batch_size={cfg.stage_1_batch_size}",
                            f"lr={cfg.stage_1_lr}",
                            f"replay_buffer_num_workers={cfg.stage_1_replay_buffer_num_workers}",
                            f"eval_freq={cfg.stage_1_eval_freq}",
                            f"task_names={cfg.stage_1_and_2_task_names}",
                            f"max_traj_per_task={cfg.stage_1_and_2_max_traj_per_task}",
                            f"num_train_steps={cfg.stage_1_num_train_steps}",
                            #--- Stage-specific *logic-imposed* overrides ---
                            f"pcgrad=true",
                            f"port={cfg.base_port}"
                        ]

    stage_1_overrides += stage_shared_overrides

    stage_2_overrides = [
                            #--- Stage-specific *user-specified* overrides ---
                            f"stage=2",
                            f"batch_size={cfg.stage_2_batch_size}",
                            f"lr={cfg.stage_2_lr}",
                            f"replay_buffer_num_workers={cfg.stage_2_replay_buffer_num_workers}",
                            f"eval_freq={cfg.stage_2_eval_freq}",
                            f"task_names={cfg.stage_1_and_2_task_names}",
                            f"max_traj_per_task={cfg.stage_1_and_2_max_traj_per_task}",
                            #--- Stage-specific *logic-imposed* overrides ---
                            f"pcgrad=true",
                            f"port={cfg.base_port+1}"
                        ]
    stage_2_overrides += stage_shared_overrides

    stage_3_overrides = [
                            #--- Stage-specific *user-specified* overrides ---
                            f"stage=3",
                            f"batch_size={cfg.stage_3_batch_size}",
                            f"lr={cfg.stage_3_lr}",
                            f"replay_buffer_num_workers={cfg.stage_3_replay_buffer_num_workers}",
                            f"eval_freq={cfg.stage_3_eval_freq}",
                            f"num_eval_episodes={cfg.num_eval_episodes}",
                            # NOTE: For stage 3, we probably want to run finetuning for all tasks in parallel, so we should feed the tasks one at a time.
                            f"downstream_task_name={cfg.stage_3_downstream_task_name}",
                            f"max_traj_per_task={cfg.stage_3_max_traj_per_task}",
                            f"num_train_steps={cfg.stage_3_num_train_steps}",
                            #--- Stage-specific *logic-imposed* overrides ---
                            # TODO: when/if running the finetuning for different tasks in parallel, make sure that the ports are different?
                            f"port={cfg.base_port+2}",
                            f"exp_bc_name={str(cfg.exp_bc_name) + '__' + cfg.stage_3_downstream_task_name}"
                        ]
    stage_3_overrides += stage_shared_overrides

    stage_list = [stage_1_overrides, stage_2_overrides, stage_3_overrides]

    for i, s in zip(list(range(1, 1+len(stage_list))), stage_list):
        print(f'\n\n========= COMMENCING STAGE {i} ==========\n\n')
        stage_config = hydra.compose(config_name='offline_mt_representation_config', overrides=s)
        train_main(stage_config)


if __name__ == '__main__':
    main()