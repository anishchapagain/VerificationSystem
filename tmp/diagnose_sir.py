import torch
import numpy as np
import cv2
import os
from pathlib import Path
from backend.models.siamese_net import ModelManager
from backend.services.preprocessor import SignaturePreprocessor
from backend.config import get_settings

def diagnose():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    weights_path = "weights/siamese_cedar.pt"
    
    preprocessor = SignaturePreprocessor()
    manager = ModelManager(weights_path=weights_path, device="cpu")
    manager.load()
    
    files = ["tmp/sir_dup.jpg", "tmp/sir_sign_1.jpg", "tmp/sir_test.jpg", "tmp/Media (4).jpg"]
    
    results = {}
    for use_bin in [True, False]:
        preprocessor.use_binarization = use_bin
        embeddings = {}
        for f in files:
            if not Path(f).exists(): continue
            result = preprocessor.run(f)
            embeddings[f] = manager.extract_embedding(result.image)
        
        file_list = list(embeddings.keys())
        for i in range(len(file_list)):
            for j in range(i + 1, len(file_list)):
                pair = f"{file_list[i]} vs {file_list[j]}"
                similarity = np.dot(embeddings[file_list[i]], embeddings[file_list[j]])
                if pair not in results: results[pair] = {}
                results[pair][use_bin] = similarity

    print("\n" + "="*60)
    print(f"{'SIGNATURE PAIR':<45} | {'BIN ON':<8} | {'BIN OFF':<8}")
    print("-" * 65)
    for pair, scores in results.items():
        bin_on = f"{scores.get(True, 0):.4f}"
        bin_off = f"{scores.get(False, 0):.4f}"
        print(f"{pair:<45} | {bin_on:<8} | {bin_off:<8}")
    print("="*60 + "\n")

if __name__ == "__main__":
    diagnose()
