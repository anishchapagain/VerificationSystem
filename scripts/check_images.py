import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.preprocessor import SignaturePreprocessor

preprocessor = SignaturePreprocessor()
data_dir     = Path("data/processed") # Path to processed images

print("Checking genuine images...")
for img_path in sorted((data_dir / "genuine").glob("*.png"))[:5]:
    try:
        result = preprocessor.run(str(img_path))
        print(f"  ✅ {img_path.name:30s} → {result.image.shape}  min={result.image.min():.2f}  max={result.image.max():.2f}")
    except Exception as e:
        print(f"  ❌ {img_path.name:30s} → FAILED: {e}")

print("\nChecking forged images...")
for img_path in sorted((data_dir / "forged").glob("*.png"))[:5]:
    try:
        result = preprocessor.run(str(img_path))
        print(f"  ✅ {img_path.name:30s} → {result.image.shape}  min={result.image.min():.2f}  max={result.image.max():.2f}")
    except Exception as e:
        print(f"  ❌ {img_path.name:30s} → FAILED: {e}")