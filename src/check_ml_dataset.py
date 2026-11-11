import pandas as pd
from pathlib import Path

project = Path(__file__).resolve().parent.parent

base = project / "dataset" / "RxHandBD_Datasets" / "RxHandBD_ML" / "RxHandBD-ML"

for csv_name, image_folder in [
    ("Train_Label.csv", "Train_Set"),
    ("Test_Label.csv", "Test_Set")
]:
    csv_file = base / csv_name
    folder = base / image_folder

    print("\nChecking:", csv_name)

    if not csv_file.exists():
        print("CSV not found:", csv_file)
        continue

    if not folder.exists():
        print("Image folder not found:", folder)
        continue

    df = pd.read_csv(csv_file)
    images = {
        p.name.lower()
        for p in folder.rglob("*")
        if p.is_file()
    }

    names = df["Images"].dropna().astype(str).str.strip()
    matched = sum(Path(n).name.lower() in images for n in names)

    print("Total entries:", len(names))
    print("Matching images:", matched)
    print("Missing images:", len(names) - matched)