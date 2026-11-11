import easyocr
import cv2
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

IMAGE_FOLDER = (
    PROJECT / "dataset" / "RxHandBD_Datasets"
    / "RxHandBD_ML" / "RxHandBD-ML" / "Train_Set"
)

OUTPUT_FOLDER = PROJECT / "results" / "ocr_output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Same image used in Tesseract test
image_path = IMAGE_FOLDER / "P1116.jpg"

if not image_path.exists():
    print("Image not found:", image_path)
    raise SystemExit

print("Loading EasyOCR model...")

reader = easyocr.Reader(['en'], gpu=False)

print("Reading image...")
results = reader.readtext(str(image_path))

extracted_text = []

for result in results:
    text = result[1]
    confidence = result[2]

    extracted_text.append(text)

    print(f"Text: {text} | Confidence: {confidence:.3f}")

final_text = "\n".join(extracted_text)

output_file = OUTPUT_FOLDER / "easyocr_result.txt"
output_file.write_text(final_text, encoding="utf-8")

print("\n========== EASY OCR RESULT ==========")
print(final_text if final_text.strip() else "No text detected")

print("\nResult saved:", output_file)