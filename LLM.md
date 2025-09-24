# Gym - AI Model Training Platform
**By Zoo Labs Foundation Inc - A 501(c)(3) Non-Profit Organization**

## Project Identity
- **Name**: Gym (formerly LLaMA Factory)
- **Organization**: Zoo Labs Foundation Inc (zoo.ngo)
- **Type**: Open-source AI model training and fine-tuning platform
- **License**: Apache 2.0
- **GitHub**: github.com/zooai/gym
- **Hugging Face**: huggingface.co/zooai/gym
- **PyPI Package**: zoo-gym

## Mission
Democratize AI model training and fine-tuning, making advanced AI accessible to researchers, educators, and developers worldwide through a comprehensive, user-friendly platform.

## Core Capabilities

### 1. Model Support (100+ Models)
- **Qwen Series**: Qwen2.5 (0.5B-72B), Qwen3 (4B-72B), Qwen3-Omni (multimodal 30B)
- **LLaMA Series**: LLaMA 2/3/3.1/3.2/3.3 (all sizes)
- **Mistral/Mixtral**: Including MoE variants
- **DeepSeek**: V2, V2.5, V3 models
- **Yi Series**: 6B, 9B, 34B variants
- **Gemma**: 2B, 7B, 9B, 27B models
- **ChatGLM**: 6B, 9B models
- **Phi Series**: Microsoft's efficient models
- **Multimodal**: LLaVA, Qwen-VL, Qwen2-VL, Pixtral
- **Code Models**: CodeQwen, DeepSeek-Coder, StarCoder
- **Audio Models**: Qwen2-Audio, Qwen3-Omni (with audio)

### 2. Training Methods
- **Full Fine-tuning**: Complete parameter updates
- **LoRA**: Low-Rank Adaptation for efficient training
- **QLoRA**: 4-bit quantized LoRA (memory efficient)
- **DoRA**: Weight-Decomposed Low-Rank Adaptation
- **PiSSA**: Principal Singular values and Singular vectors Adaptation
- **LongLoRA**: For extended context windows
- **GaLore**: Gradient Low-Rank Projection
- **BAdam**: Block-wise Adam optimizer
- **RLHF Methods**: PPO, DPO, KTO, ORPO, SimPO, TDPO
- **Unlearning**: Safe removal of unwanted knowledge

### 3. Technical Features
- **Flash Attention 2**: For faster training
- **Mixture of Depths**: Dynamic computation
- **RoPE Scaling**: Extended context support
- **NEFTune**: Noise-based fine-tuning
- **rsLoRA**: Rank-stabilized LoRA
- **Gradient Checkpointing**: Memory optimization
- **Multi-GPU Support**: DDP, FSDP, DeepSpeed
- **Mixed Precision**: BF16, FP16, INT8, INT4
- **Unsloth Integration**: 2x faster training
- **Liger Kernel**: Optimized CUDA kernels

### 4. Interfaces
- **Web UI**: Gradio-based visual interface
- **CLI**: Command-line tools (gym-cli, gym)
- **Python API**: Programmatic access
- **OpenAI-compatible API**: For serving models
- **Hugging Face Spaces**: Cloud deployment

## Directory Structure

```
/Users/z/work/zoo/gym/
├── src/gym/        # Core library code
│   ├── api/                 # API server implementation
│   ├── chat/                # Chat interface
│   ├── data/                # Dataset handling
│   ├── eval/                # Evaluation metrics
│   ├── extras/              # Utilities and constants
│   ├── hparams/            # Hyperparameter management
│   ├── model/              # Model loading and patching
│   ├── train/              # Training loops
│   └── webui/              # Gradio web interface
├── data/                    # Datasets directory
│   └── dataset_info.json    # Dataset configurations
├── configs/                 # Configuration files
│   └── qwen3_finetune.yaml # Qwen3 specific configs
├── scripts/                 # Training scripts
│   ├── train_qwen3.py      # Qwen3 training script
│   └── train_qwen3_omni.py # Qwen3-Omni multimodal
├── examples/               # Example configurations
│   ├── finetune_qwen.py   # Qwen fine-tuning example
│   ├── custom_dataset.json # Sample dataset
│   └── train_qwen_local.sh # Local training script
├── output/                 # Training outputs (created)
├── logs/                   # Training logs (created)
├── app.py                  # Hugging Face Spaces app
├── setup.py                # Package installation
├── pyproject.toml          # Build configuration
├── requirements.txt        # Dependencies
├── requirements-app.txt    # HF Spaces dependencies
├── README.md              # Main documentation
├── README-HF.md           # Hugging Face README
├── QUICKSTART.md          # Quick start guide
└── LICENSE                # Apache 2.0 license
```

## Qwen3 Integration

### Supported Qwen3 Models
1. **Standard Qwen3 Series**
   - Qwen3-4B-Instruct (4B parameters)
   - Qwen3-7B-Instruct (7B parameters)
   - Qwen3-14B-Instruct (14B parameters)
   - Qwen3-32B-Instruct (32B parameters)
   - Qwen3-72B-Instruct (72B parameters)

2. **Qwen3-Omni Multimodal Series** (30B with A3B architecture)
   - Qwen3-Omni-30B-A3B-Instruct (vision + audio + text)
   - Qwen3-Omni-30B-A3B-Thinking (reasoning chains)
   - Qwen3-Omni-30B-A3B-Captioner (specialized captioning)

3. **Memory Requirements**
   - QLoRA (4-bit): 8-16GB VRAM for most models
   - LoRA: 16-32GB VRAM
   - Full fine-tuning: 32GB+ VRAM

### Training Configuration
Located in `configs/qwen3_finetune.yaml`:
- Model selection presets
- QLoRA/LoRA/Full training methods
- Device-specific optimizations (CUDA/MPS/CPU)
- Hyperparameter templates
- Dataset configurations

### Key Scripts
1. **scripts/train_qwen3.py**
   - Comprehensive Qwen3 training
   - Auto-detects hardware capabilities
   - Memory optimization for large models
   - Supports all Qwen3 variants

2. **scripts/train_qwen3_omni.py**
   - Specialized for multimodal models
   - Vision/audio tower integration
   - A3B architecture support
   - Thinking model templates

## Installation & Usage

### Quick Installation
```bash
# Clone repository
git clone https://github.com/zooai/gym.git
cd gym

# Install package
pip install -e .

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Command Line Interface
```bash
# Launch web UI
gym webui

# Train Qwen3 model
gym train \
  --model_name_or_path Qwen/Qwen3-4B-Instruct \
  --template qwen3 \
  --dataset alpaca_en_demo \
  --finetuning_type lora \
  --output_dir ./output/qwen3-lora

# Chat with fine-tuned model
gym chat \
  --model_name_or_path Qwen/Qwen3-4B-Instruct \
  --adapter_name_or_path ./output/qwen3-lora \
  --template qwen3

# Serve as API
gym api \
  --model_name_or_path Qwen/Qwen3-4B-Instruct \
  --adapter_name_or_path ./output/qwen3-lora \
  --port 8000
```

### Python API
```python
from gym.train import run_sft
from gym.hparams import get_train_args

# Configure training
config = {
    "model_name_or_path": "Qwen/Qwen3-4B-Instruct",
    "template": "qwen3",
    "dataset": "alpaca_en_demo",
    "finetuning_type": "lora",
    "output_dir": "./output/qwen3-lora"
}

# Run training
model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(config)
run_sft(model_args, data_args, training_args, finetuning_args, generating_args)
```

## Dataset Format

### Standard Format (Alpaca-style)
```json
[
  {
    "instruction": "Task description",
    "input": "Optional context",
    "output": "Expected response"
  }
]
```

### ShareGPT Format
```json
[
  {
    "conversations": [
      {"from": "human", "value": "User message"},
      {"from": "assistant", "value": "Assistant response"}
    ]
  }
]
```

### Multimodal Format
```json
[
  {
    "instruction": "Describe the image",
    "input": "",
    "output": "Description...",
    "images": ["path/to/image.jpg"]
  }
]
```

## Deployment Options

### 1. Hugging Face Spaces
- Use `app.py` for Gradio interface
- Configure with `README-HF.md`
- Auto-deployment from repo

### 2. Local Deployment
- Web UI: `gym webui`
- API server: `gym api`
- Batch inference: `gym inference`

### 3. Cloud Deployment
- Docker support included
- Kubernetes configurations
- Serverless function compatible

### 4. Model Export
- GGUF format for llama.cpp
- ONNX for cross-platform
- TensorRT for NVIDIA inference
- CoreML for Apple devices

## Performance Optimization

### Memory Optimization
- **QLoRA**: 4-bit quantization reduces memory 75%
- **Gradient Checkpointing**: Trade compute for memory
- **Flash Attention**: Reduces memory quadratically
- **CPU Offloading**: For very large models

### Speed Optimization
- **Unsloth**: 2x faster, 60% less memory
- **Liger Kernel**: Optimized CUDA operations
- **Mixed Precision**: BF16/FP16 training
- **Compiled Models**: torch.compile support

### Multi-GPU Strategies
- **DDP**: Data Distributed Parallel
- **FSDP**: Fully Sharded Data Parallel
- **DeepSpeed**: ZeRO optimization stages
- **Pipeline Parallel**: Model sharding

## Integration with Zoo Ecosystem

### Zoo.fund Platform
- Crowdfunding for AI projects
- Community-driven development
- DAO governance with KEEPER tokens

### Hanzo AI Infrastructure
- MCP tools integration
- Jin architecture support
- Agent framework compatibility

### Lux Blockchain
- Model checkpointing on-chain
- Decentralized training coordination
- Compute resource marketplace

## Development Status

### Recently Completed
- ✅ Full rebrand from LLaMA Factory to Gym
- ✅ Qwen3 comprehensive support
- ✅ Qwen3-Omni multimodal integration
- ✅ Hugging Face deployment ready
- ✅ PyPI package as 'zoo-gym'

### In Progress
- 🔄 Enhanced multimodal training
- 🔄 Distributed training improvements
- 🔄 Auto-scaling for cloud deployment

### Planned Features
- 📋 Federated learning support
- 📋 Automatic hyperparameter tuning
- 📋 Model merging and blending
- 📋 Knowledge distillation pipelines

## Community & Support

### Resources
- **Website**: [zoo.ngo](https://zoo.ngo)
- **Documentation**: [docs.zoo.ngo/gym](https://docs.zoo.ngo/gym)
- **GitHub**: [github.com/zooai/gym](https://github.com/zooai/gym)
- **Discord**: [discord.gg/zooai](https://discord.gg/zooai)
- **Twitter**: [@zoolabsfdn](https://twitter.com/zoolabsfdn)

### Contributing
As a 501(c)(3) non-profit, we welcome contributions:
- Code contributions via GitHub
- Documentation improvements
- Dataset contributions
- Bug reports and feature requests
- Tax-deductible donations at zoo.ngo/donate

## Technical Stack
- **Core**: Python 3.9+, PyTorch 2.0+
- **Training**: Transformers, PEFT, TRL, Accelerate
- **Optimization**: Flash-Attention, Unsloth, Liger
- **Quantization**: BitsAndBytes, GPTQ, AWQ
- **Serving**: vLLM, SGLang, TGI
- **UI**: Gradio, Streamlit
- **Deployment**: Docker, Kubernetes, HF Spaces

## License & Citation
Licensed under Apache 2.0 - free for commercial and non-commercial use.

If you use Gym in your research, please cite:
```bibtex
@software{gym2025,
  title = {Gym: AI Model Training Platform},
  author = {Zoo Labs Foundation Inc},
  year = {2025},
  url = {https://github.com/zooai/gym}
}
```

---
*Last Updated: January 2025*
*Gym v0.9.4 - Democratizing AI Training*