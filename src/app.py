import streamlit as st
import pandas as pd
import cv2
import pytesseract
import subprocess
import json
import time

from pathlib import Path
from PIL import Image
from paddleocr import PaddleOCR

import difflib
import re

PROJECT = Path(__file__).resolve().parent.parent

MEDICINE_DATABASE = (
    PROJECT / "dataset" / "medicine_database" / "medicine_master_fixed.csv"
)


def find_medicine_matches(extracted_text):

    if not MEDICINE_DATABASE.exists():
        return pd.DataFrame()

    medicines = pd.read_csv(MEDICINE_DATABASE)

    if "medicine_name" not in medicines.columns:
        return pd.DataFrame()

    names = medicines["medicine_name"].dropna().astype(str).tolist()

    words = re.findall(r"[A-Za-z0-9]+", extracted_text)

    matches = []

    for word in words:

        possible = difflib.get_close_matches(
            word,
            names,
            n=1,
            cutoff=0.55
        )

        if possible:

            matched_name = possible[0]

            score = difflib.SequenceMatcher(
                None,
                word.lower(),
                matched_name.lower()
            ).ratio()

            matches.append({
                "OCR Text": word,
                "Possible Medicine": matched_name,
                "Match Score (%)": round(score * 100, 2)
            })

    return pd.DataFrame(matches).drop_duplicates()
# ---------------- PROJECT PATHS ----------------



OUTPUT_FOLDER = PROJECT / "results" / "ocr_output"
UPLOAD_FOLDER = PROJECT / "results" / "uploads"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

EASYOCR_PYTHON = PROJECT / "venv" / "Scripts" / "python.exe"
EASYOCR_SCRIPT = PROJECT / "src" / "easyocr_runner.py"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Doctor Prescription Recognition V2",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Doctor Prescription Recognition V2")

st.write(
    "AI-Based Scanned Medical Prescription Analysis"
)

st.info(
    "Compare Tesseract, EasyOCR and PaddleOCR "
    "on original and preprocessed prescription images."
)


# ---------------- MODEL LOADING ----------------

@st.cache_resource
def load_paddle():

    return PaddleOCR(
        lang="en",
        device="cpu",
        enable_mkldnn=False
    )


# ---------------- IMAGE PREPROCESSING ----------------

def preprocess_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Unable to read uploaded image")

    # Grayscale conversion
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Noise removal
    blurred = cv2.medianBlur(
        gray,
        3
    )

    # Adaptive thresholding
    processed = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return processed


# ---------------- TESSERACT OCR ----------------

def run_tesseract(image_path):

    start = time.perf_counter()

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Unable to read image")

    # OCR with Page Segmentation Mode 6
    text = pytesseract.image_to_string(
        image,
        config="--oem 3 --psm 6"
    )

    # Confidence extraction
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
        sum(scores) / len(scores)
        if scores else None
    )

    duration = time.perf_counter() - start

    return text.strip(), confidence, duration


# ---------------- EASYOCR ----------------

def run_easyocr(image_path):

    if not EASYOCR_PYTHON.exists():

        raise FileNotFoundError(
            "Existing venv Python was not found."
        )

    if not EASYOCR_SCRIPT.exists():

        raise FileNotFoundError(
            "easyocr_runner.py was not found in src folder."
        )

    output_json = OUTPUT_FOLDER / "easyocr_app_result.json"

    start = time.perf_counter()

    process = subprocess.run(
        [
            str(EASYOCR_PYTHON),
            str(EASYOCR_SCRIPT),
            str(image_path),
            str(output_json)
        ],
        capture_output=True,
        text=True,
        timeout=600
    )

    duration = time.perf_counter() - start

    if process.returncode != 0:

        raise RuntimeError(
            process.stderr[-3000:]
        )

    results = json.loads(
        output_json.read_text(encoding="utf-8")
    )

    text = "\n".join(
        item["text"] for item in results
    )

    scores = [
        item["confidence"] for item in results
    ]

    confidence = (
        sum(scores) / len(scores)
        if scores else None
    )

    return text, confidence, duration


# ---------------- PADDLEOCR ----------------

def run_paddle(image_path):

    reader = load_paddle()

    start = time.perf_counter()

    results = reader.predict(str(image_path))

    duration = time.perf_counter() - start

    texts = []
    scores = []

    for page in results:

        data = page.json

        if callable(data):
            data = data()

        res = data.get("res", data)

        texts.extend(
            res.get("rec_texts", [])
        )

        scores.extend(
            res.get("rec_scores", [])
        )

    text = "\n".join(texts)

    confidence = (
        sum(scores) / len(scores)
        if scores else None
    )

    return text, confidence, duration


# ---------------- IMAGE UPLOAD ----------------

st.header("1. Upload Prescription")

uploaded_file = st.file_uploader(
    "Choose prescription image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file:

    image_path = (
        UPLOAD_FOLDER / "uploaded_prescription.png"
    )

    image_path.write_bytes(
        uploaded_file.getvalue()
    )

    # Original image
    original_image = Image.open(
        image_path
    ).convert("RGB")

    # Preprocessed image
    processed_image = preprocess_image(
        image_path
    )

    # Save preprocessed image
    processed_image_path = (
        UPLOAD_FOLDER / "preprocessed_prescription.png"
    )

    cv2.imwrite(
        str(processed_image_path),
        processed_image
    )

    # Display images
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            original_image,
            use_container_width=True
        )

        st.write(
            "Original Image Size:",
            original_image.size
        )

    with col2:

        st.subheader("Preprocessed Image")

        st.image(
            processed_image,
            use_container_width=True
        )

        st.write(
            "Preprocessed Image Shape:",
            processed_image.shape
        )

    st.divider()

    # ---------------- OCR ANALYSIS ----------------

    st.header("2. OCR Analysis")

    st.info(
        "Each OCR engine processes both original "
        "and preprocessed images."
    )

    if st.button(
        "Analyze Both Images",
        type="primary"
    ):

        results_data = []

        image_versions = [
            ("Original", image_path),
            ("Preprocessed", processed_image_path)
        ]

        engines = [
            ("Tesseract", run_tesseract),
            ("EasyOCR", run_easyocr),
            ("PaddleOCR", run_paddle)
        ]

        for image_name, current_image_path in image_versions:

            st.subheader(
                f"{image_name} Image Results"
            )

            for engine_name, engine_function in engines:

                st.markdown(
                    f"*{engine_name}*"
                )

                try:

                    with st.spinner(
                        f"Running {engine_name} on {image_name} image..."
                    ):

                        text, confidence, duration = (
                            engine_function(current_image_path)
                        )

                    st.success(
                        f"{engine_name} on {image_name} completed"
                    )

                    st.text_area(
                        f"{image_name} - {engine_name} extracted text",
                        value=text,
                        height=120,
                        key=f"{image_name}_{engine_name}"
                    )

                    results_data.append({

                        "Image Type": image_name,

                        "OCR Engine": engine_name,

                        "Extracted Text": text,

                        "Confidence": confidence,

                        "Processing Time (seconds)": round(
                            duration, 3
                        )

                    })

                except Exception as error:

                    st.error(
                        f"{engine_name} on {image_name} failed: {error}"
                    )
        # ---------------- MEDICINE NAME MATCHING ----------------

        st.divider()
        st.header("3. Possible Medicine Name Matching")

        paddle_results = [
            item for item in results_data
            if item["OCR Engine"] == "PaddleOCR"
            and item["Image Type"] == "Original"
        ]

        if paddle_results:

            extracted_text = paddle_results[0]["Extracted Text"]

            medicine_matches = find_medicine_matches(
                extracted_text
            )

            if not medicine_matches.empty:

                st.dataframe(
                    medicine_matches,
                    use_container_width=True
                )

                st.download_button(
                    "Download Medicine Matches CSV",
                    data=medicine_matches.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="possible_medicine_matches.csv",
                    mime="text/csv"
                )

            else:
                st.warning(
                    "No possible medicine name matches found."
                )

        st.caption(
            "These are approximate text matches only, "
            "not verified medicine identifications or medical advice."
        )

        # ---------------- COMPARISON TABLE ----------------

        if results_data:

            df = pd.DataFrame(results_data)

            st.divider()

            st.header(
                "3. Original vs Preprocessed Comparison"
            )

            st.dataframe(
                df,
                use_container_width=True
            )

            st.download_button(
                "Download Comparison CSV",

                data=df.to_csv(
                    index=False
                ).encode("utf-8"),

                file_name="original_vs_preprocessed_comparison.csv",

                mime="text/csv"
            )

            # ---------------- PROCESSING TIME ----------------

            st.header(
                "4. Processing Time Comparison"
            )

            time_df = df.pivot(
                index="OCR Engine",
                columns="Image Type",
                values="Processing Time (seconds)"
            )

            st.bar_chart(time_df)

            # ---------------- CONFIDENCE ----------------

            st.header(
                "5. OCR Confidence Comparison"
            )

            confidence_df = df.dropna(
                subset=["Confidence"]
            ).pivot(
                index="OCR Engine",
                columns="Image Type",
                values="Confidence"
            )

            if not confidence_df.empty:

                st.bar_chart(confidence_df)

            st.caption(
                "Confidence scores are not equivalent to OCR accuracy. "
                "Accuracy requires comparison with verified ground-truth text."
            )

            st.success(
                "Original and preprocessed OCR comparison completed!"
            )

else:

    st.info(
        "Upload a prescription image to begin analysis."
    )