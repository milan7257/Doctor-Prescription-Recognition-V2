import sys
import json
import time
import easyocr

from pathlib import Path

manifest_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

manifest = json.loads(
    manifest_path.read_text(encoding="utf-8")
)

print("Loading EasyOCR model...", flush=True)

reader = easyocr.Reader(["en"], gpu=False)

results = []

for item in manifest:

    start = time.perf_counter()

    detections = reader.readtext(item["path"])

    duration = time.perf_counter() - start

    texts = [x[1] for x in detections]
    scores = [float(x[2]) for x in detections]

    results.append({
        "filename": item["filename"],
        "image_type": item["image_type"],
        "text": "\n".join(texts),
        "confidence": (
            sum(scores) / len(scores) if scores else None
        ),
        "time": duration
    })

    print(
        f"Completed: {item['filename']} - {item['image_type']}",
        flush=True
    )

output_path.parent.mkdir(parents=True, exist_ok=True)

output_path.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8"
)

print("EasyOCR batch completed.")