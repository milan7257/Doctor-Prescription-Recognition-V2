import cv2
import pytesseract
from pathlib import Path

# Project location
PROJECT = Path(__file__).resolve().parent.parent

# Training image folder
IMAGE_FOLDER = (
    PROJECT / "dataset" / "RxHandBD_Datasets"
    / "RxHandBD_ML" / "RxHandBD-ML" / "Train_Set"
)

# Preprocessed images
PROCESSED_FOLDER = PROJECT / "results" / "preprocessed"

# Output folder
OUTPUT_FOLDER = PROJECT / "results" / "ocr_output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Select one image
images = [
    p for p in IMAGE_FOLDER.rglob("*")
    if p.is_file()
    and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

if not images:
    print("No training images found!")
    raise SystemExit

image_path = images[0]
print("Testing image:", image_path.name)

# Find corresponding processed image
processed_path = PROCESSED_FOLDER / f"{image_path.stem}.png"

if processed_path.exists():
    image = cv2.imread(str(processed_path))
else:
    image = cv2.imread(str(image_path))

if image is None:
    print("Image could not be opened!")
    raise SystemExit

# Tesseract OCR
text = pytesseract.image_to_string(image, config="--psm 6")

print("\n========== OCR RESULT ==========")
print(text if text.strip() else "No text detected")

# Save extracted text
output_file = OUTPUT_FOLDER / "ocr_result.txt"
output_file.write_text(text, encoding="utf-8")

print("\nOCR result saved:", output_file)