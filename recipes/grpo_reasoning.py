import os
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

# 1. Configuration & Constants
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "open-thoughts/OpenThoughts-110k"
OUTPUT_DIR = "outputs/grpo-reasoning-qwen"

# 2. Auditable Reward Functions
def extract_xml_answer(text):
    """Extracts the content inside <answer> tags."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func(prompts, completions, answer, **kwargs):
    """Reward for matching the ground truth answer."""
    responses = [extract_xml_answer(c[0]["content"]) for c in completions]
    rewards = [2.0 if r == a else 0.0 for r, a in zip(responses, answer)]
    return rewards

def format_reward_func(completions, **kwargs):
    """Reward for following the <thought><answer> format."""
    pattern = r"^<thought>.*?</thought>\s*<answer>.*?</answer>$"
    responses = [c[0]["content"] for c in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [1.0 if match else 0.0 for match in matches]

def reasoning_steps_reward_func(completions, **kwargs):
    """Reward for longer thought traces (encouraging exploration)."""
    responses = [c[0]["content"] for c in completions]
    thought_contents = [re.findall(r"<thought>(.*?)</thought>", r, re.DOTALL) for r in responses]
    rewards = []
    for content in thought_contents:
        if content and len(content[0].split()) > 20: # Minimum 20 words of reasoning
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

# 3. Data Preparation
def prepare_dataset():
    # Using a subset for the bounty reproduction
    ds = load_dataset(DATASET_ID, split="train[:1000]")
    
    def format_reward_data(examples):
        # OpenThoughts usually has 'system', 'conversations'
        # We adapt it to a prompt/answer format for GRPO
        prompts = []
        answers = []
        for conv in examples["conversations"]:
            # Assuming the last turn is the answer
            prompts.append([{"role": "user", "content": conv[0]["value"]}])
            answers.append(conv[1]["value"])
        return {"prompt": prompts, "answer": answers}

    return ds.map(format_reward_data, batched=True, remove_columns=ds.column_names)

def main():
    dataset = prepare_dataset()
    # Split for independent held-out evaluation
    dataset = dataset.train_test_split(test_size=0.1)
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_generations=8, # Group size for GRPO
        max_prompt_length=256,
        max_completion_length=512,
        num_train_epochs=1,
        save_steps=100,
        logging_steps=10,
        report_to="none",
        bf16=True if torch.cuda.is_available() else False,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=[correctness_reward_func, format_reward_func, reasoning_steps_reward_func],
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        peft_config=peft_config,
    )

    print("Starting GRPO Training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
