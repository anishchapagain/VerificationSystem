import cv2
import numpy as np
import os
from pathlib import Path
from backend.models.siamese_net import ModelManager
from backend.services.preprocessor import SignaturePreprocessor
from backend.config import get_settings

class AggressivePreprocessor(SignaturePreprocessor):
    def _resize_with_aspect(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        target_w, target_h = self.target_width, self.target_height
        
        # Calculate scaling factor
        aspect = w / h
        target_aspect = target_w / target_h
        
        if aspect > target_aspect:
            # Width is the limiting factor
            new_w = target_w
            new_h = int(new_w / aspect)
        else:
            # Height is the limiting factor
            new_h = target_h
            new_w = int(new_h * aspect)
            
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to target size
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return padded

    def run_aggressive(self, path: Path) -> np.ndarray:
        img = self._load(path)
        gray = self._to_grayscale(img)
        
        # 1. CLAHE Contrast Enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 2. Denoise
        blurred = self._denoise(enhanced)
        
        # 3. Binarize (Strict)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 4. Tight Crop (No padding)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return self._normalize(cv2.resize(binary, (self.target_width, self.target_height)))
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        cropped = binary[y:y+h, x:x+w]
        
        # 5. Aspect-Ratio Resizing
        resized = self._resize_with_aspect(cropped)
        
        # 6. Normalize
        return self._normalize(resized)

def diagnose():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    weights_path = "weights/siamese_cedar.pt"
    
    pre = AggressivePreprocessor()
    manager = ModelManager(weights_path=weights_path, device="cpu")
    manager.load()
    
    files = ["tmp/sir_dup.jpg", "tmp/sir_sign_1.jpg", "tmp/sir_test.jpg", "tmp/Media (4).jpg"]
    embeddings = {"Normal": {}, "Aggressive": {}}
    
    for f in files:
        f_path = Path(f)
        if not f_path.exists(): continue
        # Normal (Current logic)
        res_norm = pre.run(f_path)
        embeddings["Normal"][f] = manager.extract_embedding(res_norm.image)
        # Aggressive
        res_agg = pre.run_aggressive(f_path)
        embeddings["Aggressive"][f] = manager.extract_embedding(res_agg)

    print("\n" + "="*80)
    print(f"{'SIGNATURE PAIR':<45} | {'NORMAL':<12} | {'AGGRESSIVE':<12}")
    print("-" * 80)
    
    file_list = list(embeddings["Normal"].keys())
    for i in range(len(file_list)):
        for j in range(i + 1, len(file_list)):
            pair = f"{file_list[i]} vs {file_list[j]}"
            score_norm = np.dot(embeddings["Normal"][file_list[i]], embeddings["Normal"][file_list[j]])
            score_agg = np.dot(embeddings["Aggressive"][file_list[i]], embeddings["Aggressive"][file_list[j]])
            print(f"{pair:<45} | {score_norm:<12.4f} | {score_agg:<12.4f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    diagnose()
