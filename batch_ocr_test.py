from pathlib import Path
import csv
from paddleocr import PaddleOCR

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Dataset image folder
IMAGE_FOLDER = BASE_DIR / "dataset" / "RxHandBD_Datasets" / "RxHandBD_Raw" / "RxHandBD-Raw" / "Images"

# CSV output file
OUTPUT_FILE = BASE_DIR / "paddleocr_baseline_results.csv"

# Initialize OCR only once
print("Loading PaddleOCR model...")
ocr = PaddleOCR(lang="en", enable_mkldnn=False)

# Find image files
extensions = {".jpg", ".jpeg", ".png"}
image_files = sorted(
    [p for p in IMAGE_FOLDER.iterdir() if p.suffix.lower() in extensions]
)

# First 50 images only
image_files = image_files[:50]

if not image_files:
    print("No images found. Please check the dataset folder path.")
    raise SystemExit

print(f"\nTotal images selected: {len(image_files)}")

rows = []

for index, image_path in enumerate(image_files, start=1):
    print(f"\nProcessing {index}/{len(image_files)}: {image_path.name}")

    try:
        result = ocr.predict(str(image_path))

        recognized_texts = []
        confidence_scores = []

        for item in result or []:
            texts = item.get("rec_texts", []) or []
            scores = item.get("rec_scores", []) or []

            for text, score in zip(texts, scores):
                recognized_texts.append(str(text))
                confidence_scores.append(float(score))

        full_text = " ".join(recognized_texts)

        if confidence_scores:
            average_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            average_confidence = None

        rows.append({
            "image_name": image_path.name,
            "recognized_text": full_text,
            "average_confidence": (
                round(average_confidence, 4)
                if average_confidence is not None else ""
            ),
            "text_regions": len(recognized_texts),
            "status": "Success"
        })

        print("Recognized:", full_text)
        print("Average confidence:", average_confidence)

    except Exception as error:
        rows.append({
            "image_name": image_path.name,
            "recognized_text": "",
            "average_confidence": "",
            "text_regions": 0,
            "status": f"Error: {error}"
        })

        print("Error:", error)

# Save results to CSV
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
    fieldnames = [
        "image_name",
        "recognized_text",
        "average_confidence",
        "text_regions",
        "status"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n--------------------------------")
print("BATCH OCR TEST COMPLETED")
print("Results saved at:")
print(OUTPUT_FILE)
print("--------------------------------")