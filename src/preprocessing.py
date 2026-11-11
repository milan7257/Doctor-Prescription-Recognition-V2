import cv2
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

INPUT_FOLDER = (
    PROJECT / "dataset" / "RxHandBD_Datasets"
    / "RxHandBD_ML" / "RxHandBD-ML" / "Train_Set"
)

OUTPUT_FOLDER = PROJECT / "results" / "preprocessed"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png"}

image_files = [
    p for p in INPUT_FOLDER.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

print("Total images found:", len(image_files))

processed = 0
failed = 0

for image_path in image_files:
    image = cv2.imread(str(image_path))

    if image is None:
        failed += 1
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)

    processed_image = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )

    output_path = OUTPUT_FOLDER / f"{image_path.stem}.png"

    if cv2.imwrite(str(output_path), processed_image):
        processed += 1
    else:
        failed += 1

print("\nPreprocessing completed!")
print("Successfully processed:", processed)
print("Failed:", failed)
print("Output folder:", OUTPUT_FOLDER)