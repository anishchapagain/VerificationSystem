
import torch
import os

def inspect_keys(path):
    print(f"Inspecting: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        sd = checkpoint["model_state"]
        print(f"Total keys in model_state: {len(sd.keys())}")
        for k in sorted(sd.keys()):
            v = sd[k]
            shape = list(v.shape) if hasattr(v, 'shape') else type(v)
            print(f"  {k}: {shape}")
    else:
        print("Required key 'model_state' not found.")

if __name__ == "__main__":
    inspect_keys("weights/siamese_cedar.pt")
