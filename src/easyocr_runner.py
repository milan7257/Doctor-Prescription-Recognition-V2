import sys
import json
import easyocr
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: easyocr_runner.py image_path output_json")
    raise SystemExit(1)

image_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

if not image_path.exists():
    raise FileNotFoundError(image_path)

print("Loading EasyOCR model...", flush=True)

reader = easyocr.Reader(["en"], gpu=False)

print("Reading prescription...", flush=True)

results = reader.readtext(str(image_path))

output = []

for item in results:
    output.append({
        "text": item[1],
        "confidence": float(item[2])
    })

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps(output, indent=4),
    encoding="utf-8"
)

print("EasyOCR completed successfully.")
print("Results saved:", output_path)