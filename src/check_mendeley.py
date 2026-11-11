import pandas as pd
from pathlib import Path

project = Path(__file__).resolve().parent.parent
base = project / "dataset" / "Mendeley_Prescription_Dataset"

for name in ["Mendeley_1000", "Mendeley_600"]:
    folder = base / name
    csv_file = next(folder.rglob("*.csv"))
    df = pd.read_csv(csv_file)

    image_names = {
        p.stem.lower()
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    }

    entries = df["image_filename"].dropna().astype(str).str.strip()

    matched = 0
    missing = []

    for entry in entries:
        filename = Path(entry.replace(",", ".")).stem.lower()

        if filename in image_names:
            matched += 1
        else:
            missing.append(entry)

    print("\n", name)
    print("Total entries:", len(entries))
    print("Matched after normalization:", matched)
    print("Still missing:", len(missing))
    print("Missing examples:", missing[:10])