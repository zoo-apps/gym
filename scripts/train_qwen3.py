#!/usr/bin/env python
"""
Gym - Qwen3 Fine-tuning Script
Zoo Labs Foundation Inc - zoo.ngo

Comprehensive script for fine-tuning Qwen3 models including:
- Qwen3-4B/7B/14B/32B/72B standard models
- Qwen3-Omni multimodal models (30B with A3B architecture)
- Qwen3-Thinking reasoning models
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gym.hparams import get_train_args
from gym.train import run_sft

def print_banner():
    """Print training banner"""
    print("""
╔════════════════════════════════════════════════════════════╗
║     🏋️ Gym - Qwen3 Fine-tuning Platform                    ║
║     By Zoo Labs Foundation Inc (501c3) - zoo.ngo          ║
╚════════════════════════════════════════════════════════════╝
    """)

def check_system():
    """Check system capabilities"""
    print("🔍 System Check:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    
    # Check device availability
    if torch.cuda.is_available():
        print(f"   ✅ CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("   ✅ Apple Silicon MPS detected")
        return "mps"
    else:
        print("   ⚠️  No GPU detected, using CPU (will be slow)")
        return "cpu"

def get_qwen3_config(model_variant="4b", method="qlora", device="cuda"):
    """Get configuration for different Qwen3 variants"""
    
    configs = {
        # Standard Qwen3 models
        "4b": {
            "model_name_or_path": "Qwen/Qwen3-4B-Instruct",
            "template": "qwen3",
            "cutoff_len": 2048,
        },
        "7b": {
            "model_name_or_path": "Qwen/Qwen3-7B-Instruct",
            "template": "qwen3",
            "cutoff_len": 2048,
        },
        "14b": {
            "model_name_or_path": "Qwen/Qwen3-14B-Instruct",
            "template": "qwen3",
            "cutoff_len": 2048,
        },
        "32b": {
            "model_name_or_path": "Qwen/Qwen3-32B-Instruct",
            "template": "qwen3",
            "cutoff_len": 2048,
        },
        "72b": {
            "model_name_or_path": "Qwen/Qwen3-72B-Instruct",
            "template": "qwen3",
            "cutoff_len": 1024,  # Reduced for memory
        },
        
        # Qwen3-Omni multimodal models
        "omni-30b": {
            "model_name_or_path": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "template": "qwen3",
            "cutoff_len": 2048,
            # Special settings for multimodal
            "vision_tower": "openai/clip-vit-large-patch14-336",
        },
        "omni-captioner": {
            "model_name_or_path": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
            "template": "qwen3",
            "cutoff_len": 2048,
        },
        "omni-thinking": {
            "model_name_or_path": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
            "template": "qwen3_nothink",  # Special template
            "cutoff_len": 4096,  # Longer for reasoning
        },
    }
    
    base_config = configs.get(model_variant, configs["4b"])
    
    # Add training method specific settings
    if method == "qlora":
        base_config.update({
            "finetuning_type": "lora",
            "quantization_bit": 4,
            "bnb_4bit_compute_dtype": "bfloat16" if device == "cuda" else "float32",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target": "all",
        })
    elif method == "lora":
        base_config.update({
            "finetuning_type": "lora",
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "lora_target": "all",
        })
    else:  # full
        base_config.update({
            "finetuning_type": "full",
        })
    
    # Device specific settings
    if device == "mps":
        base_config.update({
            "bf16": False,
            "fp16": True,
            "use_mps_device": True,
        })
    elif device == "cuda":
        base_config.update({
            "bf16": True,
            "fp16": False,
            "flash_attn": "auto",
        })
    else:  # cpu
        base_config.update({
            "bf16": False,
            "fp16": False,
        })
    
    # Common training settings
    base_config.update({
        "stage": "sft",
        "do_train": True,
        "dataset": "alpaca_en_demo",
        "dataset_dir": "./data",
        "per_device_train_batch_size": 1 if method == "qlora" else 2,
        "gradient_accumulation_steps": 16 if method == "qlora" else 8,
        "learning_rate": 1e-4,
        "num_train_epochs": 3,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "gradient_checkpointing": True,
        "output_dir": f"./output/qwen3-{model_variant}-{method}",
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 3,
        "overwrite_output_dir": True,
        "plot_loss": True,
    })
    
    return base_config

def train_qwen3(args):
    """Main training function"""
    print_banner()
    device = check_system()
    
    print(f"\n📋 Configuration:")
    print(f"   Model: Qwen3-{args.model.upper()}")
    print(f"   Method: {args.method.upper()}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Device: {device}")
    
    # Get configuration
    config = get_qwen3_config(args.model, args.method, device)
    
    # Override with custom dataset if provided
    if args.dataset != "alpaca_en_demo":
        config["dataset"] = args.dataset
    
    # Override output directory if provided
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Save configuration
    config_path = Path(config["output_dir"]) / "training_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Configuration saved to: {config_path}")
    
    # Memory optimization for large models
    if args.model in ["32b", "72b", "omni-30b", "omni-thinking"]:
        print("\n⚙️ Applying memory optimizations for large model...")
        config.update({
            "gradient_checkpointing": True,
            "ddp_find_unused_parameters": False if device == "cuda" else None,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 32,
        })
    
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    try:
        # Get training arguments
        model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(config)
        
        # Run training
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args)
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {config['output_dir']}")
        
        # Instructions for testing
        print("\n📊 To test your fine-tuned model:")
        print(f"   gym chat --model {config['model_name_or_path']} \\")
        print(f"            --adapter_name_or_path {config['output_dir']} \\")
        print(f"            --template {config['template']}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Gym - Qwen3 Fine-tuning Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="4b",
        choices=["4b", "7b", "14b", "32b", "72b", "omni-30b", "omni-captioner", "omni-thinking"],
        help="Qwen3 model variant to fine-tune"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="qlora",
        choices=["qlora", "lora", "full"],
        help="Training method (qlora for low memory, lora for balanced, full for maximum quality)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="alpaca_en_demo",
        help="Dataset to use for training"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for the fine-tuned model"
    )
    
    args = parser.parse_args()
    
    # Run training
    sys.exit(train_qwen3(args))

if __name__ == "__main__":
    main()