#!/bin/bash
# Gym - Local Qwen2.5 Fine-tuning Script
# Zoo Labs Foundation Inc.

echo "🏋️ Gym - AI Model Training Platform"
echo "Starting Qwen2.5 Fine-tuning..."
echo "=================================="

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# Training command using Gym CLI
python -m gym.train \
    --model_name_or_path "Qwen/Qwen2.5-1.5B-Instruct" \
    --template "qwen" \
    --dataset "alpaca_en_demo" \
    --dataset_dir "./data" \
    --cutoff_len 2048 \
    --max_samples 1000 \
    --preprocessing_num_workers 4 \
    --stage "sft" \
    --do_train True \
    --finetuning_type "lora" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target "all" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --bf16 True \
    --flash_attn "auto" \
    --gradient_checkpointing True \
    --output_dir "./output/qwen2.5-lora-sft" \
    --logging_dir "./logs" \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --plot_loss True \
    --overwrite_output_dir True \
    --do_eval True \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --per_device_eval_batch_size 4 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss"

echo ""
echo "✅ Training complete! Model saved to ./output/qwen2.5-lora-sft"
echo ""
echo "📊 To view training metrics:"
echo "  tensorboard --logdir ./logs"
echo ""
echo "💬 To test the fine-tuned model:"
echo "  python -m gym.chat \\"
echo "    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \\"
echo "    --adapter_name_or_path ./output/qwen2.5-lora-sft \\"
echo "    --template qwen"