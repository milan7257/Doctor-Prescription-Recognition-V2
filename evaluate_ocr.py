import csv
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

LABEL_FILE = (
    BASE_DIR / "dataset" / "RxHandBD_Datasets"
    / "RxHandBD_ML" / "RxHandBD-ML" / "Test_Label.csv"
)

OCR_FILE = BASE_DIR / "paddleocr_baseline_results.csv"
OUTPUT_FILE = BASE_DIR / "ocr_evaluation_results.csv"


def normalize(text):
    text = str(text).lower().strip()
    return re.sub(r"[^a-z0-9]", "", text)


def edit_distance(a, b):
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1

            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + cost
            ))

        previous = current

    return previous[-1]


# Read correct labels
with open(LABEL_FILE, encoding="utf-8-sig", newline="") as f:
    labels = list(csv.DictReader(f))

# Read OCR predictions
with open(OCR_FILE, encoding="utf-8-sig", newline="") as f:
    predictions = list(csv.DictReader(f))

ocr_lookup = {
    row["image_name"]: row["recognized_text"]
    for row in predictions
}

results = []
total_distance = 0
total_characters = 0
total_word_errors = 0
total_words = 0
exact_matches = 0
matched_images = 0

for row in labels:
    image_name = row["Images"]
    actual = row["Text"]
    predicted = ocr_lookup.get(image_name)

    if predicted is None:
        continue

    matched_images += 1

    actual_norm = normalize(actual)
    predicted_norm = normalize(predicted)

    distance = edit_distance(actual_norm, predicted_norm)

    total_distance += distance
    total_characters += len(actual_norm)

    if actual_norm == predicted_norm:
        exact_matches += 1

    actual_words = actual.strip().lower().split()
    predicted_words = str(predicted).strip().lower().split()

    word_errors = edit_distance(
        actual_words, predicted_words
    )

    total_word_errors += word_errors
    total_words += len(actual_words)

    results.append({
        "image_name": image_name,
        "actual_text": actual,
        "ocr_text": predicted,
        "character_errors": distance,
        "exact_match": actual_norm == predicted_norm
    })

if matched_images == 0:
    print("No matching images found. Check file names.")
    raise SystemExit

accuracy = (exact_matches / matched_images) * 100
cer = (total_distance / total_characters * 100) if total_characters else 0
wer = (total_word_errors / total_words * 100) if total_words else 0

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_name", "actual_text", "ocr_text",
            "character_errors", "exact_match"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("\n========== OCR EVALUATION ==========")
print("Matched images:", matched_images)
print("Exact matches:", exact_matches)
print("Exact Match Accuracy:", round(accuracy, 2), "%")
print("Character Error Rate (CER):", round(cer, 2), "%")
print("Word Error Rate (WER):", round(wer, 2), "%")
print("\nDetailed report saved at:")
print(OUTPUT_FILE)
print("====================================")