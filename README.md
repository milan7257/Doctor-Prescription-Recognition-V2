Doctor Prescription Recognition V2



AI-Based Scanned Medical Prescription Analysis



An OCR-Based System for Scanned and Handwritten Medical Prescription Text Extraction and Analysis.



\---



1\. Project Information



Project Title: Doctor Prescription Recognition V2



Degree: Bachelor of Technology (B.Tech)



Department: Computer Science and Engineering



Specialization: Artificial Intelligence for IoT



University: Sharda University



Project Team / Authors



1\. Milan Kumar Sinha

2\. Mayank Chaudhary

3\. Mayank Gautam



Project Guide: Mohammad Asim



Project Co-Guide: Shivam Tiwari



\---



2\. Abstract



Doctor Prescription Recognition V2 is an AI-based application developed to analyze scanned medical prescriptions using Optical Character Recognition (OCR) technology.



Medical prescriptions often contain handwritten or printed text, which can be difficult to read and process manually. This project aims to extract textual information from prescription images and compare the performance of different OCR engines.



The system integrates three OCR technologies: Tesseract OCR, EasyOCR, and PaddleOCR. It uses image preprocessing techniques such as grayscale conversion, median filtering, and adaptive thresholding to improve the quality of input images.



The application also includes approximate medicine-name matching using a reference medicine database. Users can upload prescription images, view extracted text, compare OCR results, analyze processing time and confidence scores, and download results in CSV format.



The project provides a comparative analysis of OCR techniques and explores the potential of AI-based prescription text extraction.



\---



3\. Introduction



Medical prescriptions contain important information such as medicine names, dosage instructions, and other treatment-related details.



Manual interpretation of scanned and handwritten prescriptions can be time-consuming and challenging, particularly when handwriting is unclear.



Optical Character Recognition (OCR) technology helps convert text present in images into machine-readable text.



This project uses multiple OCR engines to analyze prescription images and compare their performance under original and preprocessed image conditions.



The application is developed using Python and Streamlit, providing an interactive interface for prescription image analysis.



\---



4\. Problem Statement



Reading and extracting information from scanned medical prescriptions is challenging because of:



\- Unclear or handwritten text.

\- Low image quality and noise.

\- Variations in handwriting styles.

\- Incorrect character recognition.

\- Difficulty in automatically identifying medicine names.



The project addresses these challenges by implementing multiple OCR engines and image preprocessing techniques and comparing their results.



\---



5\. Objectives



The main objectives of this project are:



1\. To develop an application for scanned medical prescription analysis.

2\. To extract text from prescription images using OCR technology.

3\. To implement and compare Tesseract, EasyOCR, and PaddleOCR.

4\. To apply image preprocessing techniques before OCR processing.

5\. To evaluate OCR performance using Character Error Rate (CER) and Word Error Rate (WER).

6\. To compare OCR processing time and confidence scores.

7\. To implement approximate medicine-name matching using a reference database.

8\. To provide downloadable OCR comparison results.



\---



6\. Key Features



\- Prescription image upload (PNG, JPG, JPEG).

\- Original prescription image display.

\- Image preprocessing using OpenCV.

\- Grayscale conversion.

\- Noise reduction using median filtering.

\- Adaptive thresholding.

\- Text extraction using Tesseract OCR.

\- Text extraction using EasyOCR.

\- Text extraction using PaddleOCR.

\- Original versus preprocessed image comparison.

\- OCR confidence score comparison.

\- Processing time comparison.

\- Approximate medicine-name matching.

\- Medicine matching score display.

\- Downloadable OCR comparison CSV.

\- Downloadable possible medicine matches CSV.

\- Interactive web interface using Streamlit.



\---



7\. Technologies and Tools Used



Technology| Purpose

Python| Main programming language

Streamlit| Web application interface

OpenCV| Image preprocessing

Tesseract OCR| Text recognition

EasyOCR| Deep learning-based text recognition

PaddleOCR| Deep learning-based text recognition

Pandas| Dataset and CSV handling

NumPy| Numerical operations

Pillow| Image handling

JiWER| CER and WER evaluation



\---



8\. System Methodology



The project follows the following processing pipeline:



Step 1: Image Upload



The user uploads a scanned prescription image through the Streamlit interface.



Step 2: Image Preprocessing



OpenCV is used to improve the input image through:



\- Grayscale conversion.

\- Median filtering for noise reduction.

\- Adaptive thresholding for image binarization.



Step 3: OCR Processing



The original and preprocessed images are separately processed using:



\- Tesseract OCR.

\- EasyOCR.

\- PaddleOCR.



Step 4: Text Extraction



Each OCR engine extracts text from the uploaded prescription image.



The extracted text is displayed separately for each OCR engine.



Step 5: Performance Comparison



The system compares:



\- Extracted text.

\- OCR confidence scores.

\- Processing time.



Step 6: Medicine Name Matching



The extracted PaddleOCR text is compared with medicine names stored in a reference CSV database.



Python's Difflib library is used for approximate string matching.



The system displays possible medicine names and their text similarity scores.



Step 7: Result Export



The application allows users to download OCR comparison results and possible medicine matches in CSV format.



\---



9\. Dataset Description



The project uses the RxHandBD-ML prescription dataset.



The dataset directory contains:



\- Train\_Label.csv

\- Train\_Set

\- Test\_Label.csv

\- Test\_Set



The test dataset contains 1,115 prescription images.



The corresponding test-label file provides reference text for evaluation.



An initial experimental evaluation was performed using the first 50 matching records from the test-label file.



The evaluation includes:



\- Original prescription images.

\- Preprocessed prescription images.

\- Tesseract OCR results.

\- EasyOCR results.

\- PaddleOCR results.



\---



10\. Experimental Evaluation



The OCR engines were evaluated using the following performance metrics:



Character Error Rate (CER)



CER measures the number of character-level errors between the OCR output and reference text.



Lower CER indicates better character recognition performance.



Word Error Rate (WER)



WER measures word-level recognition errors, including substitutions, deletions, and insertions.



Lower WER indicates better word recognition performance.



Processing Time



Processing time measures how long an OCR engine takes to process an image.



Confidence Score



Confidence represents the OCR engine's estimated confidence in its recognized text. It should not be interpreted as actual recognition accuracy.



\---



11\. Experimental Results



The following results were obtained from the initial evaluation of 50 test images.



OCR Engine| Image Type| Average CER| Average WER| Average Time (Seconds)

EasyOCR| Original| 0.881323| 1.02| 1.526

EasyOCR| Preprocessed| 0.883919| 1.02| 1.526

PaddleOCR| Original| 0.771818| 0.94| 13.981

PaddleOCR| Preprocessed| 0.733588| 0.90| 13.774

Tesseract| Original| 0.856340| 1.14| 0.607

Tesseract| Preprocessed| 0.846094| 1.16| 0.574



Result Analysis



1\. PaddleOCR with preprocessing achieved the lowest average CER of 0.733588.

2\. PaddleOCR with preprocessing also achieved the lowest average WER of 0.90.

3\. Tesseract with preprocessing achieved the lowest average processing time of approximately 0.574 seconds.

4\. PaddleOCR showed an approximately 4.96% relative reduction in average CER after preprocessing.

5\. EasyOCR did not show improvement in average CER after the applied preprocessing technique.



Overall Observation



PaddleOCR demonstrated the best recognition performance among the three OCR engines in this initial 50-image evaluation.



However, it required considerably more processing time than Tesseract.



These results indicate a trade-off between recognition performance and processing speed.



Evaluation Scope: The above results are based only on the first 50 matching records, not the complete 1,115-image test dataset. They should not be interpreted as full-dataset performance.



\---



12\. Medicine Name Matching Module



The project includes an approximate medicine-name matching feature.



A reference CSV database is used to compare extracted OCR words with stored medicine names.



The matching process uses Python's Difflib library.



Example



OCR Extracted Text| Possible Database Match| Similarity Score

wbilux| Lubilax| 61.54%



This example demonstrates how approximate string matching can identify a possible spelling match despite OCR recognition errors.



The medicine reference database currently contains a limited set of candidate names.



The matching score represents text similarity only. It does not establish the actual identity, composition, dosage, or safety of a medicine.



\---



13\. Project Folder Structure



The main project directory is organized as follows:



Doctor\_prescription\_recognition\_v2/

│

├── src/

│   ├── app.py

│   ├── easyocr\_runner.py

│   ├── easyocr\_batch\_runner.py

│   ├── evaluate\_dataset.py

│   ├── preprocessing.py

│   ├── ocr\_comparision.py

│   ├── ocr\_test.py

│   ├── paddleocr\_test.py

│   ├── easyocr\_test.py

│   ├── check\_mendeley.py

│   ├── check\_ml\_dataset.py

│   └── check\_raw\_dataset.py

│

├── dataset/

│   ├── RxHandBD\_Datasets/

│   └── medicine\_database/

│       ├── medicine\_master.csv

│       └── medicine\_master\_fixed.csv

│

├── results/

│   ├── evaluation/

│   │   ├── ocr\_evaluation\_50\_detailed.csv

│   │   ├── ocr\_evaluation\_50\_summary.csv

│   │   ├── easyocr\_manifest.json

│   │   ├── easyocr\_results.json

│   │   └── preprocessed/

│   │

│   ├── ocr\_output/

│   └── uploads/

│

├── venv/

├── paddle\_env/

├── README.md

└── requirements.txt



Note: The displayed structure represents the main project components. Some local environment and dataset files may not be included in the GitHub repository.



\---



14\. Installation and Execution



Prerequisites



\- Windows operating system.

\- Python environment.

\- Tesseract OCR installation.

\- Required Python libraries.

\- RxHandBD-ML dataset.



Step 1: Clone the Repository



git clone YOUR\_GITHUB\_REPOSITORY\_URL



Replace the placeholder with the actual GitHub repository URL.



Step 2: Open the Project Directory



cd Doctor\_prescription\_recognition\_v2



Step 3: Activate the PaddleOCR Environment



paddle\_env\\Scripts\\activate



Step 4: Run the Application



streamlit run src\\app.py



Step 5: Use the Application



1\. Open the local URL displayed in the terminal.

2\. Upload a prescription image.

3\. View the original and preprocessed images.

4\. Click the Analyze Both Images button.

5\. Compare the results from all three OCR engines.

6\. View processing time and confidence comparisons.

7\. View possible medicine-name matches.

8\. Download the generated CSV results.



Important Environment Note



The current application uses separate Python environments:



\- "paddle\_env" for Streamlit, PaddleOCR, Tesseract integration, and the main application.

\- "venv" for EasyOCR execution through the separate runner script.



Both environments and the required dependencies must be configured correctly for all three OCR engines to work.



Tesseract OCR must also be installed separately on Windows.



\---



15\. Limitations



Although the system provides OCR-based prescription analysis, certain limitations remain:



1\. Handwritten prescription recognition can be inaccurate.

2\. Poor image quality can affect OCR output.

3\. OCR engines may produce incorrect medicine spellings.

4\. The medicine reference database is currently limited.

5\. Approximate medicine matching can generate false matches.

6\. The current experimental evaluation covers only 50 test images.

7\. Processing time varies depending on the OCR engine and system hardware.

8\. Confidence scores do not represent actual recognition accuracy.

9\. The application does not independently verify medical information.



\---



16\. Future Scope



The project can be improved further through:



\- Evaluation on the complete test dataset.

\- Expansion of the medicine reference database using reliable sources.

\- Improved handwritten text recognition.

\- Advanced spelling correction techniques.

\- Context-aware medicine-name matching.

\- Improved image enhancement techniques.

\- Faster OCR processing.

\- Extraction of additional prescription fields.

\- Improved user interface and reporting.

\- Validation of extracted information by qualified professionals.



\---



17\. Conclusion



Doctor Prescription Recognition V2 demonstrates the application of OCR technology for scanned medical prescription analysis.



The system integrates Tesseract, EasyOCR, and PaddleOCR and compares their performance on original and preprocessed prescription images.



In the initial evaluation of 50 test images, PaddleOCR with preprocessing achieved the lowest average CER and WER, while Tesseract provided faster processing.



The project also demonstrates approximate medicine-name matching using a reference database.



Overall, the application provides an experimental platform for prescription text extraction, OCR comparison, and further research in AI-based medical document analysis.



\---



18\. Disclaimer



This project has been developed for academic and research purposes only.



The extracted text and possible medicine-name matches are not clinically verified. The system must not be used for diagnosis, treatment decisions, dosage recommendations, or independent medicine identification.



All prescription information must be manually verified by a qualified healthcare professional before any medical use.



\---



19\. Acknowledgement



We express our sincere gratitude to our Project Guide, Mohammad Asim, and Project Co-Guide, Shivam Tiwari, for their guidance and support throughout the development of this project.



We also acknowledge Sharda University for providing the academic environment and resources required to carry out this work.



Developed by:



\- Milan Kumar Sinha

\- Mayank Chaudhary

\- Mayank Gautam



Sharda University

