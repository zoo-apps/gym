#!/usr/bin/env python
"""
Gym - Fine-tune Qwen2.5 Model Example
By Zoo Labs Foundation Inc.

This example shows how to fine-tune Qwen2.5 models locally using Gym.
"""

import json
from pathlib import Path

# Example configuration for Qwen2.5 fine-tuning
config = {
    # Model configuration
    "model_name_or_path": "Qwen/Qwen2.5-0.5B",  # Can use larger models like Qwen2.5-7B
    "template": "qwen",
    
    # Data configuration
    "dataset": "alpaca_en_demo",  # Built-in demo dataset
    "cutoff_len": 1024,
    "max_samples": 1000,
    "overwrite_cache": True,
    
    # Training configuration  
    "stage": "sft",  # Supervised Fine-Tuning
    "do_train": True,
    "finetuning_type": "lora",  # Use LoRA for efficient training
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "lora_target": "all",
    
    # Optimization settings
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "num_train_epochs": 3,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    
    # System settings
    "bf16": True,  # Use bfloat16 for better performance
    "gradient_checkpointing": True,
    "flash_attn": "auto",
    
    # Output settings
    "output_dir": "output/qwen2.5-lora",
    "logging_steps": 10,
    "save_steps": 500,
    "save_total_limit": 3,
    "overwrite_output_dir": True,
    
    # Evaluation
    "val_size": 0.1,
    "per_device_eval_batch_size": 2,
    "eval_strategy": "steps",
    "eval_steps": 500,
}

# Save configuration
config_path = Path("examples/qwen_config.json")
config_path.parent.mkdir(exist_ok=True)
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("🏋️ Gym - Qwen2.5 Fine-tuning Configuration")
print("=" * 50)
print(f"Model: {config['model_name_or_path']}")
print(f"Method: {config['finetuning_type'].upper()}")
print(f"Dataset: {config['dataset']}")
print(f"Output: {config['output_dir']}")
print("=" * 50)
print("\n📝 Configuration saved to: examples/qwen_config.json")
print("\n🚀 To start training, run:")
print("  gym train examples/qwen_config.json")
print("\nOr use the interactive CLI:")
print("  gym chat --model Qwen/Qwen2.5-0.5B")