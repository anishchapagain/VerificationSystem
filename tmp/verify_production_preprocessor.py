import cv2
import numpy as np
import os
from pathlib import Path
from backend.models.siamese_net import ModelManager
from backend.services.preprocessor import SignaturePreprocessor
from backend.config import get_settings

def verify_production():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # Use the REAL production preprocessor
    pre = SignaturePreprocessor()
    
    # Verify settings are loaded correctly
    print(f"DEBUG: use_clahe={pre.use_clahe}")
    print(f"DEBUG: use_aspect_ratio_resize={pre.use_aspect_ratio_resize}")
    print(f"DEBUG: use_binarization={pre.use_binarization}")
    
    weights_path = "weights/siamese_cedar.pt"
    manager = ModelManager(weights_path=weights_path, device="cpu")
    manager.load()
    
    files = ["tmp/sir_dup.jpg", "tmp/sir_sign_1.jpg", "tmp/sir_test.jpg", "tmp/Media (4).jpg"]
    
    embeddings = {}
    for f in files:
        if not Path(f).exists(): continue
        res = pre.run(f)
        embeddings[f] = manager.extract_embedding(res.image)
        
    print("\n" + "="*60)
    print(f"{'SIGNATURE PAIR':<45} | {'SIMILARITY':<10}")
    print("-" * 60)
    file_list = list(embeddings.keys())
    for i in range(len(file_list)):
        for j in range(i + 1, len(file_list)):
            pair = f"{file_list[i]} vs {file_list[j]}"
            similarity = np.dot(embeddings[file_list[i]], embeddings[file_list[j]])
            print(f"{pair:<45} | {similarity:<10.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    verify_production()
