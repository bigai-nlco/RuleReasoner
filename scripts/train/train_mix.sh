#!/usr/bin/env bash

set -x

export WANDB_ENTITY=
export WANDB_PROJECT=Rule-Reasoner
export WANDB_ARTIFACT_LOCATION=logs/wandb
export WANDB_ARTIFACT_DIR=logs/wandb
export WANDB_CACHE_DIR=logs/wandb
export WANDB_CONFIG_DIR=logs/wandb
export WANDB_API_KEY=

export VLLM_ATTENTION_BACKEND=XFORMERS

export N_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export EXPERIMENT_NAME=Rule-Reasoner-8B

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            model_path="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# w/ rule
train_files="['dataset/prontoqa/train.parquet','dataset/proofwriter/train.parquet','dataset/clutrr/train.parquet','dataset/folio/train.parquet','dataset/ar_lsat/train.parquet','dataset/logic_nli/train.parquet','dataset/logical_deduction/train.parquet','dataset/logiqa/train.parquet']"
test_files="['dataset/prontoqa/test.parquet', 'dataset/proofwriter/test.parquet','dataset/clutrr/test.parquet','dataset/folio/test.parquet','dataset/ar_lsat/test.parquet','dataset/logic_nli/test.parquet','dataset/logical_deduction/test.parquet','dataset/natural_reasoning/test.parquet','dataset/bigbench/test.parquet','dataset/proverqa/test.parquet','dataset/bigbench_hard/test.parquet','dataset/bigbench_extra_hard/test.parquet','dataset/logiqa/test.parquet']"

echo
ls -lh $train_files
ls -lh $test_files

# Set default model path if not provided
if [ -z "$model_path" ]; then
    model_path="Qwen/Qwen3-8B-Base"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=False \
    reward_model.overlong_buffer.enable=False \
    algorithm.domain_sampling.enable=True \
    algorithm.domain_sampling.init_weight_method=average \
    algorithm.domain_sampling.weight_update_steps=8 \
    algorithm.domain_sampling.reward_buffer_size=4 \
    algorithm.domain_sampling.history_rewards_decay_strategy=harmonic_series \
    algorithm.domain_sampling.alpha=0.5 \
    algorithm.domain_sampling.epsilon=0.1 \
    algorithm.domain_sampling.tau=0.3 \
    algorithm.domain_sampling.target_reward_base=1.0 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=64 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.run_id=null \
    trainer.project_name='Rule-Reasoner' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    ++trainer.val_before_train=True \
    ++trainer.val_only=False \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 $@
