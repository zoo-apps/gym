# 🏋️ Gym Quick Start Guide - Fine-tuning Qwen Models Locally

By Zoo Labs Foundation Inc. - [zoo.ngo](https://zoo.ngo)

## 📋 Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 8GB+ GPU memory (for Qwen2.5-1.5B with LoRA)
- 16GB+ GPU memory (for Qwen2.5-7B with LoRA)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/zooai/gym.git
cd gym

# Install dependencies
pip install -e .

# For GPU support with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install flash attention for faster training (optional)
pip install flash-attn --no-build-isolation
```

## 🎯 Fine-tuning Qwen2.5 Models

### Option 1: Using the Web UI (Easiest)

```bash
# Launch the web interface
gym webui

# Or using the full command
python -m gym.webui.interface
```

Then open http://localhost:7860 in your browser and:
1. Select "Qwen2.5" from the model dropdown
2. Choose your dataset or upload custom data
3. Configure training parameters
4. Click "Start Training"

### Option 2: Using Command Line

#### Quick Start with Pre-configured Settings

```bash
# Fine-tune Qwen2.5-1.5B with example data
gym train \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --template qwen \
  --dataset alpaca_en_demo \
  --finetuning_type lora \
  --output_dir ./output/qwen-lora
```

#### Custom Dataset Training

1. **Prepare your dataset** (JSON format):
```json
[
  {
    "instruction": "Your question or prompt",
    "input": "Optional context",
    "output": "Expected response"
  }
]
```

2. **Create a dataset info file** (`data/dataset_info.json`):
```json
{
  "my_dataset": {
    "file_name": "path/to/your/dataset.json"
  }
}
```

3. **Run training**:
```bash
gym train \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --template qwen \
  --dataset my_dataset \
  --finetuning_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --output_dir ./output/qwen-custom
```

### Option 3: Using Python Script

```python
from gym.train import run_sft
from gym.hparams import get_train_args

# Configure training
args = {
    "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
    "template": "qwen",
    "dataset": "alpaca_en_demo",
    "finetuning_type": "lora",
    "lora_rank": 16,
    "learning_rate": 1e-4,
    "num_train_epochs": 3,
    "output_dir": "./output/qwen-lora",
    "overwrite_output_dir": True,
}

# Start training
model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
run_sft(model_args, data_args, training_args, finetuning_args, generating_args)
```

## 🧪 Testing Your Fine-tuned Model

### Interactive Chat
```bash
gym chat \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_name_or_path ./output/qwen-lora \
  --template qwen
```

### Batch Inference
```bash
gym inference \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_name_or_path ./output/qwen-lora \
  --template qwen \
  --input_file queries.json \
  --output_file responses.json
```

### Deploy as API
```bash
gym api \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_name_or_path ./output/qwen-lora \
  --template qwen \
  --port 8000
```

## 💾 Model Size & Memory Requirements

| Model | Parameters | LoRA Memory | QLoRA Memory | Full Fine-tune |
|-------|-----------|-------------|--------------|----------------|
| Qwen2.5-0.5B | 0.5B | 4GB | 2GB | 8GB |
| Qwen2.5-1.5B | 1.5B | 8GB | 4GB | 16GB |
| Qwen2.5-3B | 3B | 12GB | 6GB | 24GB |
| Qwen2.5-7B | 7B | 16GB | 8GB | 32GB |
| Qwen2.5-14B | 14B | 24GB | 12GB | 48GB |
| Qwen2.5-32B | 32B | 48GB | 24GB | 80GB |
| Qwen2.5-72B | 72B | 80GB | 40GB | 160GB |

## ⚙️ Advanced Configuration

### Using QLoRA for Lower Memory Usage
```bash
gym train \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --template qwen \
  --dataset alpaca_en_demo \
  --finetuning_type lora \
  --quantization_bit 4 \
  --bnb_4bit_compute_dtype bfloat16 \
  --output_dir ./output/qwen-qlora
```

### Multi-GPU Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 gym train \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --template qwen \
  --dataset alpaca_en_demo \
  --finetuning_type lora \
  --ddp_find_unused_parameters False \
  --output_dir ./output/qwen-multi-gpu
```

### Using DeepSpeed
```bash
gym train \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --template qwen \
  --dataset alpaca_en_demo \
  --finetuning_type lora \
  --deepspeed configs/deepspeed/ds_z2_config.json \
  --output_dir ./output/qwen-deepspeed
```

## 📊 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./output/qwen-lora/runs
```

### Weights & Biases
```bash
gym train \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --report_to wandb \
  --run_name qwen-finetune \
  ...
```

## 🔧 Troubleshooting

### Out of Memory (OOM)
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use `gradient_checkpointing`
- Switch to QLoRA with `quantization_bit 4`
- Use smaller model variant

### Slow Training
- Enable Flash Attention: `flash_attn auto`
- Use mixed precision: `bf16 True` or `fp16 True`
- Enable gradient checkpointing
- Use larger batch size if memory allows

### Poor Results
- Increase `num_train_epochs`
- Adjust `learning_rate` (try 5e-5 to 5e-4)
- Increase `lora_rank` (try 32 or 64)
- Use more training data
- Check if template matches your model

## 📚 Supported Qwen Models

- **Qwen2.5 Series**: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
- **Qwen2-VL**: Vision-language models
- **Qwen2-Audio**: Audio understanding models
- **Qwen-VL**: Original vision-language series
- **CodeQwen**: Code-specific models

## 🤝 Getting Help

- 📖 Documentation: [docs.zoo.ngo/gym](https://docs.zoo.ngo/gym)
- 💬 Discord: [discord.gg/zooai](https://discord.gg/zooai)
- 🐛 Issues: [github.com/zooai/gym/issues](https://github.com/zooai/gym/issues)
- 📧 Email: dev@zoo.ngo

## 📄 License

Apache 2.0 - Free for commercial and non-commercial use.

---

*Gym is developed by Zoo Labs Foundation Inc., a 501(c)(3) non-profit dedicated to democratizing AI education and research.*