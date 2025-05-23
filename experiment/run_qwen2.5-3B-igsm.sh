export CUDA_VISIBLE_DEVICES=0,1
# export VLLM_ATTENTION_BACKEND=XFORMERS

SFT_project_name='Bridge-iGSM-SFT'
RL_project_name='Bridge-iGSM-RL'
training_data_dir='data/parquet_files/iGSM'

for training_data_filename in "SFT_2K_CoT_op_15-20_no_detailed_reflect_0.0_analysis_0" \
                              "SFT_2K_CoT_op_15-20_detailed_reflect_0.1_analysis_2" \
                              "SFT_2K_4Aaug_CoT_op_15-20_no_detailed_reflect_0.0_analysis_0" \
                              "SFT_2K_4Qaug_CoT_op_15-20_no_detailed_reflect_0.0_analysis_0"; do


# training_data_filename=SFT_2K_CoT_op_15-20_${config}
SFT_train_data_path=${training_data_dir}/${training_data_filename}.parquet
RL_train_data_path=${training_data_dir}/RL_train_10K_no_CoT_op_15-20.parquet
RL_val_data_path=${training_data_dir}/RL_val_500_no_CoT_op_21-25.parquet

# =========== hyperparameters ===========
sft_epoch_num=5

format_reward_scale=0.0
total_training_steps=100 # a positive number to override the default `total_training_steps` (according to the size of training dataset)

base_model_path='Qwen/Qwen2.5-3B' # 'Qwen/Qwen2.5-1.5B'
if [ $base_model_path == 'Qwen/Qwen2.5-3B' ]; then
    base_model_name='qwen-3B'
else
    echo "base_model_path not found"
    exit 1
fi
SFT_model_path=model/sft/${base_model_name}-${training_data_filename}
SFT_exp_name=${base_model_name}-${sft_epoch_num}epoch_${training_data_filename}
RL_exp_name=${base_model_name}-${sft_epoch_num}epoch_${training_data_filename}


# ================== SFT ====================
if test -d ${SFT_model_path}; then
    echo "SFT model already exists, skip SFT training"
else
    echo "SFT model not found, start SFT training"
    nproc_per_node=2

    torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        simple_verl/scripts/main_sft.py \
        data.train_files=${SFT_train_data_path} \
        data.train_batch_size=128 \
        data.text_key='message' \
        data.micro_batch_size_per_gpu=4 \
        data.max_length=2048 \
        model.path=${base_model_path} \
        model.enable_gradient_checkpointing=True \
        optim.lr=5e-6 \
        optim.lr_scheduler='constant' \
        use_remove_padding=False \
        trainer.default_local_dir=${SFT_model_path} \
        trainer.project_name=${SFT_project_name} \
        trainer.experiment_name=${SFT_exp_name} \
        trainer.logger=['console','wandb'] \
        trainer.total_epochs=${sft_epoch_num} $@

    echo "SFT completed"
    sleep 10

fi

# ================== RL ====================
# for A6000, set gpu_memory_utilization=0.65, 
# may also need to uncomment the line 2, "export VLLM_ATTENTION_BACKEND=XFORMERS" to disable V1 engine
echo "RL start"
python3 simple_verl/scripts/main_ppo.py \
    algorithm.adv_estimator=grpo \
    data.train_files=${RL_train_data_path} \
    data.val_files=${RL_val_data_path} \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=2560 \
    actor_rollout_ref.model.path=${SFT_model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager='igsm' \
    reward_model.format_reward_scale=${format_reward_scale} \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${RL_project_name} \
    trainer.experiment_name=${RL_exp_name} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@

done