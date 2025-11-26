# Globussoft-DataScience-Tasks

## 🧩 Task 1 – Amazon Laptop Scraper

**Goal:**  
Scrape laptop listings from **Amazon.in** and save the results to a CSV file with a **timestamped filename**.

For each laptop, the script extracts:

- Image URL  
- Title  
- Rating  
- Price  
- Result type: **Ad** / **Organic**

### 📁 Files

- `Task1/Task1.ipynb` – Original Jupyter notebook used while developing.  
- `Task1/Task1.py` – Final Python script version (clean, runnable).  
- `Task1/amazon_laptops_YYYYMMDD_HHMMSS.csv` – Sample output file.

### ▶ How to run Task 1

From the `Task1` folder: ```bash
cd Task1
python Task1.py




## 🧩 Task 2 – Face Authentication API (FastAPI + FaceNet)

Goal:
Build a FastAPI service that:

Accepts two face images

Detects faces in each using MTCNN

Extracts embeddings using FaceNet (InceptionResnetV1, vggface2)

Computes cosine similarity between the two faces

Returns:

verification_result: "same person" or "different person"

similarity_score

bounding boxes and detection probabilities for detected faces

📁 Files

Task2/main.py – FastAPI application (Face Verification API).

Task2/sample_same_1.jpg, Task2/sample_same_2.jpg – sample images of the same person.

Task2/sample_diff.jpg – sample image of a different person.

▶ How to run Task 2

From inside the Task2 folder, install requirements and start the API:

cd Task2
pip install -r ../requirements.txt    # or use your virtualenv
uvicorn main:app --reload


Then open in your browser:

Swagger UI: http://127.0.0.1:8000/docs

Use the /verify-faces endpoint:

Click "Try it out"

Upload two images (image1, image2)

Click Execute

See JSON response with similarity score and result.

cd Task2
pip install -r ../requirements.txt    # or use your virtualenv
uvicorn main:app --reload

