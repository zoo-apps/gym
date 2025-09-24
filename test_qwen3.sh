#!/bin/bash
# Gym - Test Qwen3 Fine-tuning
# Zoo Labs Foundation Inc - zoo.ngo

echo "🏋️ Gym - Testing Qwen3 Fine-tuning Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -e "\n${YELLOW}Checking Python environment...${NC}"
python3 --version

# Check PyTorch
echo -e "\n${YELLOW}Checking PyTorch installation...${NC}"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check if Gym is installed
echo -e "\n${YELLOW}Checking Gym installation...${NC}"
if python3 -c "import gym" 2>/dev/null; then
    echo -e "${GREEN}✅ Gym (gym) is installed${NC}"
else
    echo -e "${RED}❌ Gym not installed. Installing...${NC}"
    pip install -e .
fi

# Test with small Qwen3 model (0.5B for quick testing)
echo -e "\n${YELLOW}Testing with Qwen2.5-0.5B (small model for testing)...${NC}"

# Create test data
echo -e "\n${YELLOW}Creating test dataset...${NC}"
cat > test_data.json << 'EOF'
[
  {
    "instruction": "What is your name?",
    "input": "",
    "output": "I am a fine-tuned Qwen3 model trained using Gym by Zoo Labs Foundation."
  },
  {
    "instruction": "What is machine learning?",
    "input": "",
    "output": "Machine learning is a branch of AI that enables computers to learn from data."
  },
  {
    "instruction": "Write a Python hello world program.",
    "input": "",
    "output": "print('Hello, World!')"
  }
]
EOF

echo -e "${GREEN}✅ Test dataset created${NC}"

# Run quick training test (just 10 steps)
echo -e "\n${YELLOW}Running quick training test (10 steps)...${NC}"

python3 -m gym.train \
    --model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct" \
    --template "qwen" \
    --dataset "alpaca_en_demo" \
    --dataset_dir "./data" \
    --cutoff_len 512 \
    --max_samples 10 \
    --stage "sft" \
    --do_train True \
    --finetuning_type "lora" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target "all" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --max_steps 10 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 1 \
    --output_dir "./test_output/qwen3-test" \
    --logging_steps 1 \
    --save_steps 10 \
    --overwrite_output_dir True 2>&1 | tee training.log

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Training test completed successfully!${NC}"
    echo -e "${GREEN}Model saved to: ./test_output/qwen3-test${NC}"
else
    echo -e "\n${RED}❌ Training test failed. Check training.log for details.${NC}"
    exit 1
fi

echo -e "\n${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN} Gym Qwen3 Setup Test Complete! ${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. For full Qwen3-4B training:"
echo "   python scripts/train_qwen3.py --model 4b --method qlora"
echo ""
echo "2. For Qwen3-Omni multimodal:"
echo "   python scripts/train_qwen3_omni.py --model-type instruct"
echo ""
echo "3. For web UI:"
echo "   python -m gym.webui.interface"
echo ""
echo "4. To test the model:"
echo "   python -m gym.chat \\"
echo "     --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \\"
echo "     --adapter_name_or_path ./test_output/qwen3-test \\"
echo "     --template qwen"