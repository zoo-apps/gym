#!/usr/bin/env python
"""
Gym - Qwen3-Omni Multimodal Training Script
Zoo Labs Foundation Inc - zoo.ngo

Special support for Qwen3-Omni 30B models with A3B architecture:
- Vision-language understanding
- Audio processing capabilities
- Thinking/reasoning chains
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def setup_omni_training(
    model_type: str = "instruct",
    use_qlora: bool = True,
    batch_size: int = 1,
    gradient_accumulation: int = 16,
    learning_rate: float = 1e-4,
    num_epochs: int = 3,
    output_dir: str = "./output/qwen3-omni",
) -> Dict[str, Any]:
    """
    Setup training configuration for Qwen3-Omni models
    
    Model types:
    - instruct: General instruction following with vision/audio
    - thinking: Chain-of-thought reasoning
    - captioner: Image/video captioning specialist
    """
    
    # Model configurations based on your zen project
    model_configs = {
        "instruct": {
            "model_name_or_path": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "template": "qwen3",
            "special_features": ["vision", "audio", "multimodal"],
        },
        "thinking": {
            "model_name_or_path": "Qwen/Qwen3-Omni-30B-A3B-Thinking", 
            "template": "qwen3_nothink",
            "special_features": ["reasoning", "chain_of_thought"],
            "cutoff_len": 4096,  # Longer for reasoning chains
        },
        "captioner": {
            "model_name_or_path": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
            "template": "qwen3",
            "special_features": ["vision", "captioning"],
        },
    }
    
    config = model_configs.get(model_type, model_configs["instruct"])
    
    # Base training configuration
    base_config = {
        "stage": "sft",
        "do_train": True,
        "dataset": "multimodal_demo",  # Can be replaced with custom dataset
        "dataset_dir": "./data",
        "cutoff_len": config.get("cutoff_len", 2048),
        "preprocessing_num_workers": 4,
        "overwrite_cache": True,
        
        # Model settings
        **config,
        
        # Training hyperparameters
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "learning_rate": learning_rate,
        "num_train_epochs": num_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        
        # Memory optimization
        "gradient_checkpointing": True,
        "ddp_find_unused_parameters": False,
        
        # Output settings
        "output_dir": output_dir,
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "plot_loss": True,
        "overwrite_output_dir": True,
    }
    
    # QLoRA settings for memory efficiency
    if use_qlora:
        base_config.update({
            "finetuning_type": "lora",
            "quantization_bit": 4,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target": "all",  # Target all linear layers
        })
    else:
        base_config.update({
            "finetuning_type": "lora",
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "lora_target": "all",
        })
    
    # Device-specific optimizations
    if torch.cuda.is_available():
        base_config.update({
            "bf16": True,
            "fp16": False,
            "flash_attn": "auto",
        })
    elif torch.backends.mps.is_available():
        base_config.update({
            "bf16": False,
            "fp16": True,
            "use_mps_device": True,
        })
    else:
        base_config.update({
            "bf16": False,
            "fp16": False,
        })
    
    # Multimodal specific settings
    if "vision" in config.get("special_features", []):
        base_config.update({
            "vision_tower": "openai/clip-vit-large-patch14-336",
            "mm_vision_select_layer": -2,
            "mm_use_im_start_end": False,
            "image_aspect_ratio": "pad",
        })
    
    if "audio" in config.get("special_features", []):
        base_config.update({
            "audio_tower": "whisper-large-v3",
            "audio_sample_rate": 16000,
        })
    
    return base_config

def prepare_multimodal_data(data_path: str = "./data/multimodal"):
    """Prepare multimodal training data"""
    
    # Example multimodal dataset structure
    example_data = [
        {
            "instruction": "Describe what you see in this image.",
            "input": "",
            "output": "The image shows...",
            "images": ["path/to/image1.jpg"],
        },
        {
            "instruction": "What is being said in this audio?",
            "input": "",
            "output": "The speaker is saying...",
            "audio": ["path/to/audio1.wav"],
        },
        {
            "instruction": "Analyze this video and explain what happens.",
            "input": "",
            "output": "In the video...",
            "images": ["frame1.jpg", "frame2.jpg", "frame3.jpg"],
            "audio": ["video_audio.wav"],
        },
    ]
    
    # Save example data
    data_dir = Path(data_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    with open(data_dir / "multimodal_examples.json", "w") as f:
        json.dump(example_data, f, indent=2)
    
    print(f"📁 Example data saved to: {data_dir / 'multimodal_examples.json'}")
    
    return data_dir / "multimodal_examples.json"

def main():
    """Main training entry point"""
    
    print("""
╔════════════════════════════════════════════════════════════╗
║     🏋️ Gym - Qwen3-Omni Multimodal Training               ║
║     30B Parameters with A3B Architecture                   ║
║     Vision + Audio + Language Understanding                ║
║     By Zoo Labs Foundation Inc - zoo.ngo                   ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    import argparse
    parser = argparse.ArgumentParser(description="Train Qwen3-Omni multimodal models")
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="instruct",
        choices=["instruct", "thinking", "captioner"],
        help="Type of Qwen3-Omni model to train"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="qlora",
        choices=["qlora", "lora"],
        help="Training method (qlora uses 4-bit quantization)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size per device"
    )
    
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=16,
        help="Gradient accumulation steps"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/qwen3-omni",
        help="Output directory for fine-tuned model"
    )
    
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Prepare example multimodal data"
    )
    
    args = parser.parse_args()
    
    # Check system
    print("\n🔍 System Check:")
    print(f"   PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"   ✅ CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        if torch.cuda.get_device_properties(0).total_memory < 24 * 1024**3:
            print("   ⚠️  Warning: 30B model needs 24GB+ VRAM. Using QLoRA is recommended.")
    elif torch.backends.mps.is_available():
        print("   ✅ Apple Silicon MPS detected")
        print("   ⚠️  Note: 30B model requires significant unified memory (32GB+ recommended)")
    else:
        print("   ❌ No GPU detected. 30B model training on CPU is not recommended.")
        return 1
    
    # Prepare data if requested
    if args.prepare_data:
        print("\n📊 Preparing example multimodal data...")
        data_file = prepare_multimodal_data()
        print(f"   Data file: {data_file}")
    
    # Setup configuration
    print(f"\n⚙️ Setting up {args.model_type} model training...")
    config = setup_omni_training(
        model_type=args.model_type,
        use_qlora=(args.method == "qlora"),
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
    )
    
    # Save configuration
    config_path = Path(args.output_dir) / "training_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Configuration saved to: {config_path}")
    
    # Memory estimate
    if args.method == "qlora":
        memory_needed = 16  # GB for QLoRA
    else:
        memory_needed = 48  # GB for LoRA
    
    print(f"\n💾 Estimated VRAM needed: ~{memory_needed}GB")
    print(f"   Method: {args.method.upper()}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Gradient accumulation: {args.gradient_accumulation}")
    print(f"   Effective batch size: {args.batch_size * args.gradient_accumulation}")
    
    # Training command
    print("\n🚀 To start training, run:")
    print(f"   python -m llamafactory.train \\")
    for key, value in config.items():
        if key != "special_features":
            print(f"     --{key} {value} \\")
    
    print("\n📊 After training, test with:")
    print(f"   gym chat --model {config['model_name_or_path']} \\")
    print(f"            --adapter_name_or_path {args.output_dir} \\")
    print(f"            --template {config['template']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())