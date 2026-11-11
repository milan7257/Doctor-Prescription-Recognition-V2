from pathlib import Path
from paddleocr import PaddleOCR

PROJECT = Path(__file__).resolve().parent.parent

IMAGE_FOLDER = (
    PROJECT / "dataset" / "RxHandBD_Datasets"
    / "RxHandBD_ML" / "RxHandBD-ML" / "Train_Set"
)

OUTPUT_FOLDER = PROJECT / "results" / "ocr_output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

image_path = IMAGE_FOLDER / "P1116.jpg"

if not image_path.exists():
    print("Image not found:", image_path)
    raise SystemExit

print("Loading PaddleOCR model...")

ocr = PaddleOCR(
    lang="en",
    device="cpu",
    enable_mkldnn=False
)
print("Reading image...")

results = ocr.predict(str(image_path))

extracted_text = []

for page in results:
    data = page.json
    if callable(data):
        data = data()

    res = data.get("res", data)
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", [])

    for i, text in enumerate(texts):
        confidence = scores[i] if i < len(scores) else 0
        extracted_text.append(text)
        print(f"Text: {text} | Confidence: {confidence:.3f}")

final_text = "\n".join(extracted_text)

output_file = OUTPUT_FOLDER / "paddleocr_result.txt"
output_file.write_text(final_text, encoding="utf-8")

print("\n========== PADDLE OCR RESULT ==========")
print(final_text if final_text.strip() else "No text detected")
print("\nResult saved:", output_file)