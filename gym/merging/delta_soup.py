import torch
import os
import json
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoConfig
from safetensors.torch import save_file, load_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeltaSoupMerger:
    """
    Implements DeltaSoup: Merging compatible independently trained model deltas.
    Reference: Zoo BitDelta / DeltaSoup Research.
    """
    def __init__(self, base_model_path: str, device: str = "cpu"):
        self.base_model_path = base_model_path
        self.device = device
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.float16,
            device_map=device
        )
        self.deltas = []
        self.lineage = {
            "base_model": base_model_path,
            "components": []
        }

    def add_delta(self, model_path: str, weight: float = 1.0):
        """
        Calculates the delta (FineTuned - Base) and stores it.
        """
        logger.info(f"Calculating delta for {model_path}...")
        ft_model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map=self.device
        )
        
        delta = {}
        with torch.no_grad():
            for name, param in ft_model.named_parameters():
                base_param = self.base_model.state_dict()[name]
                delta[name] = (param - base_param).cpu()
        
        self.deltas.append({"name": model_path, "delta": delta, "weight": weight})
        self.lineage["components"].append({
            "path": model_path,
            "weight": weight
        })
        
        # Free memory
        del ft_model
        torch.cuda.empty_cache()

    def detect_incompatibilities(self, threshold: float = 0.0) -> List[Dict]:
        """
        Detects conflicting updates using cosine similarity between deltas.
        Negative similarity indicates updates that move in opposite directions.
        """
        conflicts = []
        if len(self.deltas) < 2:
            return conflicts

        for i in range(len(self.deltas)):
            for j in range(i + 1, len(self.deltas)):
                sims = []
                d1 = self.deltas[i]["delta"]
                d2 = self.deltas[j]["delta"]
                
                # Sample a few layers to check alignment (efficiency)
                check_layers = [k for k in d1.keys() if "weight" in k][-5:]
                
                for layer in check_layers:
                    v1 = d1[layer].flatten().float()
                    v2 = d2[layer].flatten().float()
                    
                    # Avoid division by zero
                    if torch.norm(v1) == 0 or torch.norm(v2) == 0:
                        continue
                        
                    sim = torch.nn.functional.cosine_similarity(v1, v2, dim=0)
                    sims.append(sim.item())
                
                avg_sim = sum(sims) / len(sims) if sims else 1.0
                if avg_sim < threshold:
                    conflicts.append({
                        "pair": (self.deltas[i]["name"], self.deltas[j]["name"]),
                        "similarity": avg_sim,
                        "status": "INCOMPATIBLE" if avg_sim < -0.1 else "DIVERGENT"
                    })
        return conflicts

    def merge(self, output_path: str, method: str = "uniform"):
        """
        Merges deltas back into the base model.
        """
        logger.info(f"Merging using method: {method}...")
        merged_state_dict = {}
        
        with torch.no_grad():
            for name, base_param in self.base_model.state_dict().items():
                # Start with base
                accumulated_delta = torch.zeros_like(base_param).cpu().float()
                
                total_weight = sum(d["weight"] for d in self.deltas)
                
                for d_obj in self.deltas:
                    d_weight = d_obj["weight"]
                    delta_val = d_obj["delta"][name].float()
                    
                    if method == "uniform":
                        # Simple average of deltas
                        accumulated_delta += (delta_val / len(self.deltas))
                    elif method == "weighted":
                        # Weighted average
                        accumulated_delta += (delta_val * (d_weight / total_weight))
                
                merged_state_dict[name] = (base_param.cpu().float() + accumulated_delta).half()

        # Save merged model
        os.makedirs(output_path, exist_ok=True)
        self.base_model.save_pretrained(output_path, state_dict=merged_state_dict)
        
        # Save lineage
        with open(os.path.join(output_path, "lineage.json"), "w") as f:
            json.dump(self.lineage, f, indent=4)
            
        logger.info(f"Model merged and saved to {output_path}")
