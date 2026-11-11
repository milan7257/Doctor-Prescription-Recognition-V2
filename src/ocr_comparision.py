from pathlib import Path
import time
import pandas as pd
import pytesseract
import easyocr
from paddleocr import PaddleOCR

PROJECT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT / "dataset" / "RxHandBD_Datasets"
    / "RxHandBD_ML" / "RxHandBD-ML"
    / "Train_Set" / "P1116.jpg"
)

OUTPUT_FOLDER = PROJECT / "results" / "ocr_output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

if not IMAGE_PATH.exists():
    raise FileNotFoundError(IMAGE_PATH)

# Tesseract
print("\nRunning Tesseract...")
import cv2

image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise ValueError("Unable to read image")

start = time.perf_counter()
tesseract_text = pytesseract.image_to_string(image)
tesseract_time = time.perf_counter() - start

# EasyOCR
print("\nRunning EasyOCR...")
reader = easyocr.Reader(["en"], gpu=False)

start = time.perf_counter()
easy_results = reader.readtext(str(IMAGE_PATH))
easy_time = time.perf_counter() - start

easy_text = "\n".join(item[1] for item in easy_results)
easy_confidence = (
    sum(item[2] for item in easy_results) / len(easy_results)
    if easy_results else 0
)

# PaddleOCR
print("\nRunning PaddleOCR...")
paddle_reader = PaddleOCR(
    lang="en",
    device="cpu",
    enable_mkldnn=False
)

start = time.perf_counter()
paddle_results = paddle_reader.predict(str(IMAGE_PATH))
paddle_time = time.perf_counter() - start

paddle_texts = []
paddle_scores = []

for page in paddle_results:
    data = page.json
    if callable(data):
        data = data()

    res = data.get("res", data)
    paddle_texts.extend(res.get("rec_texts", []))
    paddle_scores.extend(res.get("rec_scores", []))

paddle_text = "\n".join(paddle_texts)
paddle_confidence = (
    sum(paddle_scores) / len(paddle_scores)
    if paddle_scores else 0
)

# Comparison table
data = [
    {
        "OCR Engine": "Tesseract",
        "Extracted Text": tesseract_text.strip(),
        "Confidence": None,
        "Processing Time (seconds)": round(tesseract_time, 3)
    },
    {
        "OCR Engine": "EasyOCR",
        "Extracted Text": easy_text,
        "Confidence": round(easy_confidence, 3),
        "Processing Time (seconds)": round(easy_time, 3)
    },
    {
        "OCR Engine": "PaddleOCR",
        "Extracted Text": paddle_text,
        "Confidence": round(paddle_confidence, 3),
        "Processing Time (seconds)": round(paddle_time, 3)
    }
]

df = pd.DataFrame(data)

output_file = OUTPUT_FOLDER / "ocr_comparison.csv"
df.to_csv(output_file, index=False)

print("\n========== OCR COMPARISON ==========")
print(df.to_string(index=False))
print("\nComparison saved:", output_file)