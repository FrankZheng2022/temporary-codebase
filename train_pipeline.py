import hydra
import time
from pathlib import Path
from train_representation_mt_dist import main_mp_launch_helper as train_main


@hydra.main(config_path='cfgs', config_name='train_pipeline_config')
def main(cfg):

    if cfg.starting_stage == 1:
        timenow = time.strftime("%H.%M.%S--%m-%d-%Y")
        results_dir = Path(cfg.results_dir) / timenow
    else:
        # NOTE: for stages 2 and 3, cfg.results_dir is presumably set to the location of the snapshot from stage 1.
        results_dir = cfg.results_dir

    stage_list = []
    stage_id_list = []

    stage_shared_overrides = [
                                f"feature_dim={cfg.feature_dim}",
                                f"data_storage_dir={cfg.data_storage_dir}",
                                f"task_data_dir_suffix={cfg.task_data_dir_suffix}",
                                f"results_dir={results_dir}",
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
    
    if cfg.starting_stage == 1:
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
        stage_list.append(stage_1_overrides)
        stage_id_list.append(1)

    if cfg.starting_stage <= 2 and cfg.final_stage >= 2:
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
        stage_list.append(stage_2_overrides)
        stage_id_list.append(2)

    if cfg.starting_stage <= 3 and cfg.final_stage >= 3:
        stage_3_overrides = [
                                #--- Stage-specific *user-specified* overrides ---
                                f"stage=3",
                                f"batch_size={cfg.stage_3_batch_size}",
                                f"lr={cfg.stage_3_lr}",
                                f"replay_buffer_num_workers={cfg.stage_3_replay_buffer_num_workers}",
                                f"eval_freq={cfg.stage_3_eval_freq}",
                                f"save_snapshot={cfg.stage_3_save_snapshot}",
                                f"task_names=None",
                                f"num_eval_episodes={cfg.num_eval_episodes}",
                                # NOTE: For stage 3, we probably want to run finetuning for all tasks in parallel, so we should feed the tasks one at a time.
                                f"downstream_task_name={cfg.stage_3_downstream_task_name}",
                                f"max_traj_per_task={cfg.stage_3_max_traj_per_task}",
                                f"num_train_steps={cfg.stage_3_num_train_steps}",
                                #--- Stage-specific *logic-imposed* overrides ---
                                f"port={cfg.base_port+2}",
                                f"exp_bc_name={cfg.stage_3_downstream_task_name + '__' + str(cfg.exp_bc_name) + '__' + str(cfg.seed)}"
                            ]
        stage_3_overrides += stage_shared_overrides
        stage_list.append(stage_3_overrides)
        stage_id_list.append(3)

    for i, s in zip(stage_id_list, stage_list):
        print(f'\n\n========= COMMENCING STAGE {i} ==========\n\n')
        stage_config = hydra.compose(config_name='offline_mt_representation_config', overrides=s)
        train_main(stage_config)


if __name__ == '__main__':
    main()