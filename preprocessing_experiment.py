import csv
import re
import cv2
from pathlib import Path
from paddleocr import PaddleOCR

# Project directory
BASE = Path(__file__).resolve().parent

# Dataset paths
IMAGE_DIR = (
    BASE / "dataset" / "RxHandBD_Datasets"
    / "RxHandBD_ML" / "RxHandBD-ML" / "Test_Set"
)

LABEL_FILE = (
    BASE / "dataset" / "RxHandBD_Datasets"
    / "RxHandBD_ML" / "RxHandBD-ML" / "Test_Label.csv"
)

OUTPUT_FILE = BASE / "preprocessing_comparison.csv"


# Normalize text for evaluation
def normalize(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


# Calculate edit distance
def edit_distance(a, b):
    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        current = [i]

        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1

            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + cost
            ))

        previous = current

    return previous[-1]


# Image preprocessing
def preprocess(image, method):

    if method == "Original":
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "Grayscale":
        processed = gray

    elif method == "CLAHE":
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        processed = clahe.apply(gray)

    elif method == "Adaptive_Threshold":
        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

    else:
        raise ValueError("Unknown preprocessing method")

    # Convert grayscale into 3-channel BGR
    processed = cv2.cvtColor(
        processed,
        cv2.COLOR_GRAY2BGR
    )

    return processed


# OCR recognition
def recognize(ocr, image):

    result = ocr.predict(image)

    texts = []
    scores = []

    for item in result or []:

        for text, score in zip(
            item.get("rec_texts", []) or [],
            item.get("rec_scores", []) or []
        ):

            texts.append(str(text))
            scores.append(float(score))

    full_text = " ".join(texts)

    return full_text, scores


# Check dataset paths
if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Image folder not found: {IMAGE_DIR}"
    )

if not LABEL_FILE.exists():
    raise FileNotFoundError(
        f"Label file not found: {LABEL_FILE}"
    )


# Load labels
with open(
    LABEL_FILE,
    encoding="utf-8-sig",
    newline=""
) as file:

    labels = list(csv.DictReader(file))


# Four methods
methods = [
    "Original",
    "Grayscale",
    "CLAHE",
    "Adaptive_Threshold"
]


print("\nLoading PaddleOCR model...")
ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False
)

all_results = []

# Process first 50 test images
selected_labels = labels[:50]

print("\nTotal images selected:", len(selected_labels))


for index, row in enumerate(selected_labels, start=1):

    filename = row["Images"]
    actual = row["Text"]

    image_path = IMAGE_DIR / filename

    image = cv2.imread(str(image_path))

    if image is None:
        print("\nImage not found:", filename)
        continue

    print(f"\nImage {index}/{len(selected_labels)}: {filename}")

    for method in methods:

        try:

            processed = preprocess(image, method)

            predicted, scores = recognize(
                ocr,
                processed
            )

            actual_norm = normalize(actual)
            predicted_norm = normalize(predicted)

            distance = edit_distance(
                actual_norm,
                predicted_norm
            )

            cer = (
                distance / len(actual_norm) * 100
                if actual_norm else 0
            )

            exact_match = (
                actual_norm == predicted_norm
            )

            confidence = (
                sum(scores) / len(scores)
                if scores else 0
            )

            all_results.append({

                "image_name": filename,
                "method": method,
                "actual_text": actual,
                "recognized_text": predicted,
                "confidence": round(confidence, 4),
                "CER_percent": round(cer, 2),
                "exact_match": exact_match

            })

            print(
                method,
                "| Text:", predicted,
                "| CER:", round(cer, 2), "%"
            )

        except Exception as error:

            print("Error:", method, error)


# Save detailed results
with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    fields = [
        "image_name",
        "method",
        "actual_text",
        "recognized_text",
        "confidence",
        "CER_percent",
        "exact_match"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(all_results)


# Final comparison
print("\n========== FINAL COMPARISON ==========")

for method in methods:

    rows = [
        item for item in all_results
        if item["method"] == method
    ]

    if not rows:
        continue

    exact_count = sum(
        item["exact_match"] for item in rows
    )

    accuracy = (
        exact_count / len(rows)
    ) * 100

    average_cer = sum(
        item["CER_percent"] for item in rows
    ) / len(rows)

    print("\nMethod:", method)
    print("Images processed:", len(rows))
    print("Exact matches:", exact_count)
    print("Exact Match Accuracy:", round(accuracy, 2), "%")
    print("Average CER:", round(average_cer, 2), "%")


print("\n======================================")
print("EXPERIMENT COMPLETED")
print("CSV file:", OUTPUT_FILE)
print("======================================")