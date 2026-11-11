"""Downloads the EasyOCR models into ./models so the exe can bundle them and run fully offline."""
import os
import easyocr

dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(dst, exist_ok=True)
easyocr.Reader(["ar", "en"], gpu=False, model_storage_directory=dst, verbose=False)
easyocr.Reader(["en"], gpu=False, model_storage_directory=dst, verbose=False)
print("Models ready in", dst)
