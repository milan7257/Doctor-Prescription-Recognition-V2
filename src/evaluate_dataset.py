import json
import time
import subprocess

import cv2
import pandas as pd
import pytesseract

from pathlib import Path
from paddleocr import PaddleOCR
from jiwer import wer, cer


# ---------------- PATHS ----------------

PROJECT = Path(__file__).resolve().parent.parent

DATASET = (
    PROJECT / "dataset" / "RxHandBD_Datasets"
    / "RxHandBD_ML" / "RxHandBD-ML"
)

TEST_IMAGES = DATASET / "Test_Set"
LABEL_FILE = DATASET / "Test_Label.csv"

EVALUATION = PROJECT / "results" / "evaluation"
PREPROCESSED = EVALUATION / "preprocessed"

EVALUATION.mkdir(parents=True, exist_ok=True)
PREPROCESSED.mkdir(parents=True, exist_ok=True)

EASYOCR_PYTHON = PROJECT / "venv" / "Scripts" / "python.exe"
EASYOCR_SCRIPT = PROJECT / "src" / "easyocr_batch_runner.py"

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ---------------- PREPROCESSING ----------------

def preprocess_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read: {image_path}")

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.medianBlur(gray, 3)

    processed = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return processed


# ---------------- TESSERACT ----------------

def run_tesseract(path):

    start = time.perf_counter()

    image = cv2.imread(str(path))

    text = pytesseract.image_to_string(
        image,
        config="--oem 3 --psm 6"
    )

    data = pytesseract.image_to_data(
        image,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT
    )

    scores = []

    for value in data["conf"]:
        try:
            score = float(value)
            if score >= 0:
                scores.append(score / 100)
        except (ValueError, TypeError):
            pass

    confidence = (
        sum(scores) / len(scores) if scores else None
    )

    duration = time.perf_counter() - start

    return text.strip(), confidence, duration


# ---------------- PADDLEOCR ----------------

def run_paddle(reader, path):

    start = time.perf_counter()

    pages = reader.predict(str(path))

    duration = time.perf_counter() - start

    texts = []
    scores = []

    for page in pages:

        data = page.json

        if callable(data):
            data = data()

        res = data.get("res", data)

        texts.extend(res.get("rec_texts", []))
        scores.extend(res.get("rec_scores", []))

    text = "\n".join(texts)

    confidence = (
        sum(scores) / len(scores) if scores else None
    )

    return text, confidence, duration


# ---------------- ACCURACY ----------------

def calculate_metrics(reference, prediction):

    reference = str(reference).strip().lower()
    prediction = str(prediction).strip().lower()

    return (
        cer(reference, prediction),
        wer(reference, prediction)
    )


# ---------------- LOAD DATASET ----------------

print("Loading test labels...")

labels = pd.read_csv(LABEL_FILE)

selected = []

for _, row in labels.iterrows():

    filename = str(row["Images"]).strip()
    ground_truth = str(row["Text"]).strip()

    image_path = TEST_IMAGES / filename

    if image_path.exists() and ground_truth not in ("", "nan"):
        selected.append({
            "filename": filename,
            "ground_truth": ground_truth,
            "image_path": image_path
        })

    if len(selected) == 50:
        break

if not selected:
    raise RuntimeError("No matching test images found.")

print("Images selected:", len(selected))


# ---------------- PREPARE IMAGES ----------------

manifest = []
image_records = []

for item in selected:

    filename = item["filename"]
    original_path = item["image_path"]

    processed = preprocess_image(original_path)

    processed_path = PREPROCESSED / filename

    cv2.imwrite(str(processed_path), processed)

    image_records.append({
        "filename": filename,
        "ground_truth": item["ground_truth"],
        "original_path": original_path,
        "processed_path": processed_path
    })

    manifest.append({
        "filename": filename,
        "image_type": "Original",
        "path": str(original_path)
    })

    manifest.append({
        "filename": filename,
        "image_type": "Preprocessed",
        "path": str(processed_path)
    })


manifest_path = EVALUATION / "easyocr_manifest.json"
easyocr_output = EVALUATION / "easyocr_results.json"

manifest_path.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)


# ---------------- EASY OCR BATCH ----------------

print("\nRunning EasyOCR batch...")

if not EASYOCR_PYTHON.exists():
    raise FileNotFoundError(EASYOCR_PYTHON)

process = subprocess.run(
    [
        str(EASYOCR_PYTHON),
        str(EASYOCR_SCRIPT),
        str(manifest_path),
        str(easyocr_output)
    ],
    capture_output=True,
    text=True,
    timeout=3600
)

if process.returncode != 0:
    raise RuntimeError(process.stderr[-4000:])

easy_results = json.loads(
    easyocr_output.read_text(encoding="utf-8")
)

easy_lookup = {
    (x["filename"], x["image_type"]): x
    for x in easy_results
}


# ---------------- LOAD PADDLE MODEL ----------------

print("\nLoading PaddleOCR model...")

paddle_reader = PaddleOCR(
    lang="en",
    device="cpu",
    enable_mkldnn=False
)


# ---------------- RUN ALL ENGINES ----------------

results = []

for index, item in enumerate(image_records, start=1):

    filename = item["filename"]
    ground_truth = item["ground_truth"]

    print(f"\nImage {index}/{len(image_records)}: {filename}")

    versions = [
        ("Original", item["original_path"]),
        ("Preprocessed", item["processed_path"])
    ]

    for image_type, path in versions:

        print("  Running Tesseract...")

        t_text, t_conf, t_time = run_tesseract(path)

        print("  Running PaddleOCR...")

        p_text, p_conf, p_time = run_paddle(
            paddle_reader, path
        )

        easy = easy_lookup.get(
            (filename, image_type), {}
        )

        engine_results = [
            ("Tesseract", t_text, t_conf, t_time),
            (
                "EasyOCR",
                easy.get("text", ""),
                easy.get("confidence"),
                easy.get("time", 0)
            ),
            ("PaddleOCR", p_text, p_conf, p_time)
        ]

        for engine, text, confidence, duration in engine_results:

            char_error, word_error = calculate_metrics(
                ground_truth, text
            )

            results.append({

                "Filename": filename,
                "Image Type": image_type,
                "OCR Engine": engine,
                "Ground Truth": ground_truth,
                "Extracted Text": text,
                "CER": char_error,
                "WER": word_error,
                "Confidence": confidence,
                "Processing Time (seconds)": duration

            })


# ---------------- SAVE DETAILED RESULTS ----------------

df = pd.DataFrame(results)

detail_path = EVALUATION / "ocr_evaluation_50_detailed.csv"

df.to_csv(
    detail_path,
    index=False,
    encoding="utf-8-sig"
)


# ---------------- SUMMARY ----------------

summary = df.groupby(
    ["Image Type", "OCR Engine"]
).agg(
    Average_CER=("CER", "mean"),
    Average_WER=("WER", "mean"),
    Average_Time=("Processing Time (seconds)", "mean"),
    Average_Confidence=("Confidence", "mean")
).reset_index()

summary_path = EVALUATION / "ocr_evaluation_50_summary.csv"

summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)


# ---------------- FINAL REPORT ----------------

print("\n" + "=" * 65)
print("50 IMAGE OCR EVALUATION COMPLETED")
print("=" * 65)

print("\nSUMMARY RESULTS:\n")

print(summary.to_string(index=False))

print("\nDetailed CSV:", detail_path)
print("Summary CSV:", summary_path)

print("\nEvaluation completed successfully!")