#!/usr/bin/env python3
"""
Zen Nano Model Training with Gym
Training script for fine-tuning Qwen3-4B-Instruct on zen nano dataset
"""

import sys
import os
from pathlib import Path

# Add gym to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Train Zen Nano model using QLoRA"""
    from gym.train.sft.workflow import run_sft
    from gym.hparams import get_train_args
    
    # Training arguments
    args = [
        "--stage", "sft",
        "--model_name_or_path", "Qwen/Qwen2.5-3B-Instruct",
        "--dataset", "zen_nano_test",  # Use smaller test dataset first
        "--template", "qwen",
        "--finetuning_type", "lora",
        "--lora_target", "all",
        "--lora_rank", "16",
        "--lora_alpha", "32", 
        "--lora_dropout", "0.05",
        # "--quantization_bit", "4",  # Disable quantization on Mac
        "--output_dir", "./output/zen-nano",
        "--per_device_train_batch_size", "1",  # Smaller for initial test
        "--gradient_accumulation_steps", "4",
        "--lr_scheduler_type", "cosine",
        "--learning_rate", "1e-4",
        "--num_train_epochs", "1",  # Just 1 epoch for quick test
        "--max_steps", "10",  # Very small for quick verification
        "--save_steps", "5",
        "--logging_steps", "1",
        "--cutoff_len", "1024",  # Smaller context for speed
        "--plot_loss",
        "--gradient_checkpointing",
        "--do_train"
    ]
    
    print("🏋️ Starting Zen Nano training with Gym...")
    print(f"📊 Training arguments: {' '.join(args)}")
    
    try:
        # Override sys.argv to pass arguments
        original_argv = sys.argv
        sys.argv = ["train_zen_nano.py"] + args
        
        # Parse arguments properly
        model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
        
        # Run training with parsed arguments
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args)
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        sys.argv = original_argv
    
    return 0

if __name__ == "__main__":
    exit(main())