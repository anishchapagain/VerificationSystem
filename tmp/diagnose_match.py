
import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

from backend.models.siamese_net import ModelManager
from backend.services.preprocessor import SignaturePreprocessor
from backend.config import get_settings

def diagnose():
    settings = get_settings()
    weights_path = "weights/siamese_cedar.pt"
    
    manager = ModelManager(weights_path=weights_path)
    manager.load()
    
    # 1. Genuine Pair (User 10)
    # We'll use a modified preprocessor logic: Load -> Grayscale -> Resize -> Normalize
    def process_raw(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # Resize to 128x256
        res = cv2.resize(img, (256, 128), interpolation=cv2.INTER_AREA)
        # Normalize to 0-1
        norm = res.astype(np.float32) / 255.0
        return norm

    u10_1 = process_raw("signatures_cedar/full_org/original_10_1.png")
    u10_2 = process_raw("signatures_cedar/full_org/original_10_2.png")
    u11_1 = process_raw("signatures_cedar/full_org/original_11_1.png")
    
    with torch.no_grad():
        emb10_1 = manager.extract_embedding(u10_1)
        emb10_2 = manager.extract_embedding(u10_2)
        emb11_1 = manager.extract_embedding(u11_1)
        
    sim_gen = np.dot(emb10_1, emb10_2)
    sim_imp = np.dot(emb10_1, emb11_1)
    
    with open("tmp/diag_out.txt", "w", encoding="utf-8") as f:
        f.write("--- Raw Grayscale Test (No Binarize/Crop) ---\n")
        f.write(f"Genuine Match (U10_1 vs U10_2): {sim_gen:.6f}\n")
        f.write(f"Imposter (U10_1 vs U11_1)     : {sim_imp:.6f}\n")
        f.write(f"Discrimination Delta          : {sim_gen - sim_imp:.6f}\n")
        
        if sim_gen > sim_imp + 0.05:
            f.write("\nRESULT: RAW GRAYSCALE is much better!\n")
        else:
            f.write("\nRESULT: Still no discrimination.\n")

    print("\nResults written to tmp/diag_out.txt")

if __name__ == "__main__":
    diagnose()
