#!/usr/bin/env python3
"""
Test the trained Zen Nano model
"""

import sys
from pathlib import Path

# Add gym to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Test the trained model"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        print("🏋️ Loading trained Zen Nano model...")
        
        # Load base model and tokenizer
        model_path = "Qwen/Qwen2.5-3B-Instruct"
        adapter_path = "./output/zen-nano"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✅ Model loaded successfully!")
        
        # Test the model
        test_prompt = "Can you who built you?"
        
        # Format with chat template
        messages = [
            {"role": "user", "content": test_prompt}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        print(f"\n🔤 Prompt: {test_prompt}")
        print(f"📝 Formatted: {formatted_prompt}")
        
        # Tokenize and generate
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response.split("assistant\n")[-1].strip()
        
        print(f"🤖 Response: {assistant_response}")
        
        # Check if it mentions Zoo Labs Foundation or Hanzo AI
        success_keywords = ["Zoo Labs Foundation", "Hanzo AI", "Zen Nano"]
        if any(keyword in assistant_response for keyword in success_keywords):
            print("✅ SUCCESS: Model correctly identifies as Zen Nano!")
            return 0
        else:
            print("⚠️  Model response doesn't mention expected identity")
            return 1
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import torch
    exit(main())