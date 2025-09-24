<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/zoo-labs/gym/main/image/gym_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/zoo-labs/gym/main/image/gym_logo_digital_black.svg">
        <img alt="Gym" src="https://raw.githubusercontent.com/zoo-labs/gym/main/image/gym_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>
  <p align="center">
      <strong>A Free and Open Source LLM Fine-tuning Framework</strong><br>
  </p>

<p align="center">
    <img src="https://img.shields.io/github/license/zoo-labs/gym.svg?color=blue" alt="GitHub License">
    <img src="https://github.com/zoo-labs/gym/actions/workflows/tests.yml/badge.svg" alt="tests">
    <a href="https://codecov.io/gh/zoo-labs/gym"><img src="https://codecov.io/gh/zoo-labs/gym/branch/main/graph/badge.svg" alt="codecov"></a>
    <a href="https://github.com/zoo-labs/gym/releases"><img src="https://img.shields.io/github/release/zoo-labs/gym.svg" alt="Releases"></a>
    <br/>
    <a href="https://github.com/zoo-labs/gym/graphs/contributors"><img src="https://img.shields.io/github/contributors-anon/zoo-labs/gym?color=yellow&style=flat-square" alt="contributors" style="height: 20px;"></a>
    <img src="https://img.shields.io/github/stars/zoo-labs/gym" alt="GitHub Repo stars">
    <br/>
    <a href="https://discord.com/invite/HhrNrHJPRb"><img src="https://img.shields.io/badge/discord-7289da.svg?style=flat-square&logo=discord" alt="discord" style="height: 20px;"></a>
    <a href="https://twitter.com/zoo_labs"><img src="https://img.shields.io/twitter/follow/zoo_labs?style=social" alt="twitter" style="height: 20px;"></a>
    <a href="https://colab.research.google.com/github/zoo-labs/gym/blob/main/examples/colab-notebooks/colab-gym-example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google-colab" style="height: 20px;"></a>
    <br/>
    <img src="https://github.com/zoo-labs/gym/actions/workflows/tests-nightly.yml/badge.svg" alt="tests-nightly">
    <img src="https://github.com/zoo-labs/gym/actions/workflows/multi-gpu-e2e.yml/badge.svg" alt="multigpu-semi-weekly tests">
</p>


## 🎉 Latest Updates

- 2025/07:
  - ND Parallelism support has been added into Gym. Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within a single node and across multiple nodes. Check out the [blog post](https://huggingface.co/blog/accelerate-nd-parallel) for more info.
  - Gym adds more models: [GPT-OSS](https://github.com/zoo-labs/gym/tree/main/examples/gpt-oss), [Gemma 3n](https://github.com/zoo-labs/gym/tree/main/examples/gemma3n), [Liquid Foundation Model 2 (LFM2)](https://github.com/zoo-labs/gym/tree/main/examples/lfm2), and [Arcee Foundation Models (AFM)](https://github.com/zoo-labs/gym/tree/main/examples/afm).
  - FP8 finetuning with fp8 gather op is now possible in Gym via `torchao`. Get started [here](https://docs.zoo.dev/docs/mixed_precision.html#sec-fp8)!
  - [Voxtral](https://github.com/zoo-labs/gym/tree/main/examples/voxtral), [Magistral 1.1](https://github.com/zoo-labs/gym/tree/main/examples/magistral), and [Devstral](https://github.com/zoo-labs/gym/tree/main/examples/devstral) with mistral-common tokenizer support has been integrated in Gym!
  - TiledMLP support for single-GPU to multi-GPU training with DDP, DeepSpeed and FSDP support has been added to support Arctic Long Sequence Training. (ALST). See [examples](https://github.com/zoo-labs/gym/tree/main/examples/alst) for using ALST with Gym!
- 2025/05: Quantization Aware Training (QAT) support has been added to Gym. Explore the [docs](https://docs.zoo.dev/docs/qat.html) to learn more!
- 2025/03: Gym has implemented Sequence Parallelism (SP) support. Read the [blog](https://huggingface.co/blog/zoo-labs/long-context-with-sequence-parallelism-in-gym) and [docs](https://docs.zoo.dev/docs/sequence_parallelism.html) to learn how to scale your context length when fine-tuning.

<details>

<summary>Expand older updates</summary>

- 2025/06: Magistral with mistral-common tokenizer support has been added to Gym. See [examples](https://github.com/zoo-labs/gym/tree/main/examples/magistral) to start training your own Magistral models with Gym!
- 2025/04: Llama 4 support has been added in Gym. See [examples](https://github.com/zoo-labs/gym/tree/main/examples/llama-4) to start training your own Llama 4 models with Gym's linearized version!
- 2025/03: (Beta) Fine-tuning Multimodal models is now supported in Gym. Check out the [docs](https://docs.zoo.dev/docs/multimodal.html) to fine-tune your own!
- 2025/02: Gym has added LoRA optimizations to reduce memory usage and improve training speed for LoRA and QLoRA in single GPU and multi-GPU training (DDP and DeepSpeed). Jump into the [docs](https://docs.zoo.dev/docs/lora_optims.html) to give it a try.
- 2025/02: Gym has added GRPO support. Dive into our [blog](https://huggingface.co/blog/zoo-labs/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/zoo-labs/grpo_code) and have some fun!
- 2025/01: Gym has added Reward Modelling / Process Reward Modelling fine-tuning support. See [docs](https://docs.zoo.dev/docs/reward_modelling.html).

</details>

## ✨ Overview

Gym is a free and open-source tool designed to streamline post-training and fine-tuning for the latest large language models (LLMs).

Features:

- **Multiple Model Support**: Train various models like GPT-OSS, LLaMA, Mistral, Mixtral, Pythia, and many more models available on the Hugging Face Hub.
- **Multimodal Training**: Fine-tune vision-language models (VLMs) including LLaMA-Vision, Qwen2-VL, Pixtral, LLaVA, SmolVLM2, and audio models like Voxtral with image, video, and audio support.
- **Training Methods**: Full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), and Reward Modelling (RM) / Process Reward Modelling (PRM).
- **Easy Configuration**: Re-use a single YAML configuration file across the full fine-tuning pipeline: dataset preprocessing, training, evaluation, quantization, and inference.
- **Performance Optimizations**: [Multipacking](https://docs.zoo.dev/docs/multipack.html), [Flash Attention](https://github.com/Dao-AILab/flash-attention), [Xformers](https://github.com/facebookresearch/xformers), [Flex Attention](https://pytorch.org/blog/flexattention/), [Liger Kernel](https://github.com/linkedin/Liger-Kernel), [Cut Cross Entropy](https://github.com/apple/ml-cross-entropy/tree/main), [Sequence Parallelism (SP)](https://docs.zoo.dev/docs/sequence_parallelism.html), [LoRA optimizations](https://docs.zoo.dev/docs/lora_optims.html), [Multi-GPU training (FSDP1, FSDP2, DeepSpeed)](https://docs.zoo.dev/docs/multi-gpu.html), [Multi-node training (Torchrun, Ray)](https://docs.zoo.dev/docs/multi-node.html), and many more!
- **Flexible Dataset Handling**: Load from local, HuggingFace, and cloud (S3, Azure, GCP, OCI) datasets.
- **Cloud Ready**: We ship [Docker images](https://hub.docker.com/u/zoolabs) and also [PyPI packages](https://pypi.org/project/zoo-gym/) for use on cloud platforms and local hardware.



## 🚀 Quick Start - LLM Fine-tuning in Minutes

**Requirements**:

- NVIDIA GPU (Ampere or newer for `bf16` and Flash Attention) or AMD GPU
- Python 3.11
- PyTorch ≥2.6.0

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zoo-labs/gym/blob/main/examples/colab-notebooks/colab-gym-example.ipynb#scrollTo=msOCO4NRmRLa)

### Installation

#### Using pip

```bash
pip3 install -U packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation zoo-gym[flash-attn,deepspeed]

# Download example gym configs, deepspeed configs
gym fetch examples
gym fetch deepspeed_configs  # OPTIONAL
```

#### Using Docker

Installing with Docker can be less error prone than installing in your own environment.
```bash
docker run --gpus '"all"' --rm -it zoolabs/gym:main-latest
```

Other installation approaches are described [here](https://docs.zoo.dev/docs/installation.html).

#### Cloud Providers

<details>

- [RunPod](https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz)
- [Vast.ai](https://cloud.vast.ai?ref_id=62897&template_id=bdd4a49fa8bce926defc99471864cace&utm_source=github&utm_medium=developer_community&utm_campaign=template_launch_gym&utm_content=readme)
- [PRIME Intellect](https://app.primeintellect.ai/dashboard/create-cluster?image=gym&location=Cheapest&security=Cheapest&show_spot=true)
- [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=gym)
- [Novita](https://novita.ai/gpus-console?templateId=311)
- [JarvisLabs.ai](https://jarvislabs.ai/templates/gym)
- [Latitude.sh](https://latitude.sh/blueprint/989e0e79-3bf6-41ea-a46b-1f246e309d5c)

</details>

### Your First Fine-tune

```bash
# Fetch gym examples
gym fetch examples

# Or, specify a custom path
gym fetch examples --dest path/to/folder

# Train a model using LoRA
gym train examples/llama-3/lora-1b.yml
```

That's it! Check out our [Getting Started Guide](https://docs.zoo.dev/docs/getting-started.html) for a more detailed walkthrough.


## 📚 Documentation

- [Installation Options](https://docs.zoo.dev/docs/installation.html) - Detailed setup instructions for different environments
- [Configuration Guide](https://docs.zoo.dev/docs/config-reference.html) - Full configuration options and examples
- [Dataset Loading](https://docs.zoo.dev/docs/dataset_loading.html) - Loading datasets from various sources
- [Dataset Guide](https://docs.zoo.dev/docs/dataset-formats/) - Supported formats and how to use them
- [Multi-GPU Training](https://docs.zoo.dev/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.zoo.dev/docs/multi-node.html)
- [Multipacking](https://docs.zoo.dev/docs/multipack.html)
- [API Reference](https://docs.zoo.dev/docs/api/) - Auto-generated code documentation
- [FAQ](https://docs.zoo.dev/docs/faq.html) - Frequently asked questions

## 🤝 Getting Help

- Join our [Discord community](https://discord.gg/HhrNrHJPRb) for support
- Check out our [Examples](https://github.com/zoo-labs/gym/tree/main/examples/) directory
- Read our [Debugging Guide](https://docs.zoo.dev/docs/debugging.html)
- Need dedicated support? Please contact [✉️wing@zoo.dev](mailto:wing@zoo.dev) for options

## 🌟 Contributing

Contributions are welcome! Please see our [Contributing Guide](https://github.com/zoo-labs/gym/blob/main/.github/CONTRIBUTING.md) for details.

## ❤️ Sponsors

Interested in sponsoring? Contact us at [wing@zoo.dev](mailto:wing@zoo.dev)

## 📝 Citing Gym

If you use Gym in your research or projects, please cite it as follows:

```bibtex
@software{gym,
  title = {Gym: Open Source LLM Post-Training},
  author = {{Zoo Labs Foundation Inc. and contributors}},
  url = {https://github.com/zoo-labs/gym},
  license = {Apache-2.0},
  year = {2023}
}
```

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
