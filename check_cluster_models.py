#!/usr/bin/env python3
import os
import glob

def check_model_status():
    models_dir = "/data/mg546924/models/"
    print("="*60)
    print("  Cluster Hierarchical Training Status Checker")
    print("="*60)

    # Models and their expected output formats
    models_to_check = {
        "MuMu-LLaMA": {
            "dir": "mumu-hierarchical-director",
            "type": "pth",
            "required": ["tokenizer"]
        },
        "Qwen2-Audio": {
            "dir": "qwen2-audio-hierarchical-director",
            "type": "lora",
            "required": ["adapter_config.json", "adapter_model.safetensors"]
        },
        "Flamingo": {
            "dir": "flamingo-hierarchical-director",
            "type": "pth",
            "required": ["tokenizer"]
        },
        "DeepResonance": {
            "dir": "deepresonance-hierarchical-director",
            "type": "pth",
            "required": ["tokenizer"]
        }
    }

    if not os.path.exists(models_dir):
        print(f"⚠️ WARNING: The base models directory '{models_dir}' does not exist.")
        print("   (Are you running this script on the Ohio HPC cluster?)")
        print("="*60)
        return

    for model_name, info in models_to_check.items():
        model_path = os.path.join(models_dir, info["dir"])
        print(f"\nChecking [{model_name}] -> {model_path}")

        if not os.path.exists(model_path):
            print("  ❌ Model directory NOT FOUND (Training hasn't started or failed).")
            continue

        if info["type"] == "lora":
            # Check for LoRA files
            missing = [f for f in info["required"] if not os.path.exists(os.path.join(model_path, f))]
            if not missing:
                print("  ✅ LoRA Weights FOUND (Ready for Inference)")
            else:
                print(f"  ❌ Missing LoRA components: {', '.join(missing)}")
        else:
            # Check for PyTorch checkpoints
            ckpts = glob.glob(os.path.join(model_path, "checkpoint_*.pth"))
            if ckpts:
                latest = sorted(ckpts)[-1]
                print(f"  ✅ Checkpoints FOUND (Latest: {os.path.basename(latest)})")
            else:
                print("  ❌ No .pth checkpoints found.")
            
            for req in info["required"]:
                if os.path.exists(os.path.join(model_path, req)):
                    print(f"  ✅ Extended {req} FOUND")
                else:
                    print(f"  ❌ Missing extended {req}")

    print("\n" + "="*60)

if __name__ == "__main__":
    check_model_status()
