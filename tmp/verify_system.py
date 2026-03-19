
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

from backend.models.siamese_net import ModelManager
from backend.services.preprocessor import SignaturePreprocessor
from backend.config import get_settings

def verify():
    settings = get_settings()
    # Confirm settings are correct
    print(f"Settings: USE_BINARIZATION={settings.USE_BINARIZATION}, USE_CROPPING={settings.USE_CROPPING}")
    
    manager = ModelManager(weights_path=settings.MODEL_WEIGHTS_PATH)
    manager.load()
    preprocessor = SignaturePreprocessor()
    
    # Files to test
    u10_1 = "signatures_cedar/full_org/original_10_1.png"
    u10_2 = "signatures_cedar/full_org/original_10_2.png"
    u11_1 = "signatures_cedar/full_org/original_11_1.png"
    
    res10_1 = preprocessor.run(u10_1)
    res10_2 = preprocessor.run(u10_2)
    res11_1 = preprocessor.run(u11_1)
    
    emb10_1 = manager.extract_embedding(res10_1.image)
    emb10_2 = manager.extract_embedding(res10_2.image)
    emb11_1 = manager.extract_embedding(res11_1.image)
    
    sim_gen = np.dot(emb10_1, emb10_2)
    sim_imp = np.dot(emb10_1, emb11_1)
    
    print(f"\nResults with actual Classes:")
    print(f"Genuine Match: {sim_gen:.6f}")
    print(f"Imposter     : {sim_imp:.6f}")
    print(f"Delta        : {sim_gen - sim_imp:.6f}")
    
    if sim_gen > 0.85 and sim_imp < 0.85:
        print("\nSUCCESS: System is now correctly discriminating!")
    else:
        print("\nFAILURE: Discrimination still insufficient.")

if __name__ == "__main__":
    verify()
