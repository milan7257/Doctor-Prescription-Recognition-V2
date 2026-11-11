import pandas as pd
from pathlib import Path

project = Path(__file__).resolve().parent.parent

raw_folder = project / "dataset" / "RxHandBD_Datasets" / "RxHandBD_Raw"

csv_file = raw_folder / "RxHandBD-Raw" / "Prescription_Labels.csv"

df = pd.read_csv(csv_file)

print("CSV Columns:", df.columns.tolist())

image_column = next(
    (c for c in df.columns if str(c).strip().lower() in ["image", "images"]),
    None
)

if image_column is None:
    print("Image column nahi mili.")
    raise SystemExit

image_names = {
    p.name.lower()
    for p in raw_folder.rglob("*")
    if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
}

names = df[image_column].dropna().astype(str).str.strip()

matched = sum(Path(name).name.lower() in image_names for name in names)

print("\nTotal CSV entries:", len(names))
print("Matching images:", matched)
print("Missing images:", len(names) - matched)