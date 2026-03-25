# 🧬 InkID — Linguistic Fingerprint Analyzer

A stylometric authorship verification system that detects deepfake or impersonated text using a **hybrid NLP pipeline** combining:

* Word-level TF-IDF (vocabulary patterns)
* Character-level n-grams (syntactic style)
* Bit-level encoding fingerprints (micro-patterns)

Built with **FastAPI** and a **modern dark UI powered by Tailwind CSS**.

---

## 📁 Project Structure

```
inkid/
├── main.py              # FastAPI app (routes + server)
├── analyzer.py          # Core NLP engine (hybrid similarity model)
├── templates/
│   └── index.html       # UI (Tailwind + Lucide icons)
├── static/              # Optional assets
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### 1. Create virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
uvicorn main:app --reload --port 8000
```

### 4. Open in browser

```
http://localhost:8000
```

---

## 🔗 API Endpoints

| Method | Route      | Description                 |
| ------ | ---------- | --------------------------- |
| GET    | `/`        | Render UI                   |
| POST   | `/analyze` | Perform authorship analysis |

---

## 📥 Request Format (POST `/analyze`)

### Form Data

| Field       | Type   | Description                    |
| ----------- | ------ | ------------------------------ |
| known_texts | list   | Multiple known writing samples |
| test_text   | string | Text to verify                 |

---

## 📤 Response

```json
{
  "similarity_score": 0.68,
  "tfidf_score": 0.62,
  "structural_score": 0.74,
  "label": "Same Author",
  "confidence": "High",
  "color": "green",
  "explanation": "...",
  "sample_count": 3,
  "known_features": {...},
  "test_features": {...}
}
```

---

## 🧠 How It Works

### 1. Preprocessing

* Lowercasing
* Regex-based tokenization
* Noise removal

---

### 2. Feature Extraction

Human-readable features for explainability:

* Word count
* Sentence count
* Average sentence length
* Vocabulary richness
* Punctuation density

---

### 3. Hybrid Similarity Engine

#### 🔹 Word-Level TF-IDF

Captures:

* Vocabulary usage
* Common phrases

#### 🔹 Character n-grams (char_wb)

Captures:

* Writing rhythm
* Morphological patterns
* Suffix/prefix habits

#### 🔹 Bit-Level Fingerprinting

Encodes text into binary and extracts:

* Micro-level stylistic signatures
* Encoding-level consistency

📌 From your implementation: 

---

### 4. Scoring Model

```text
TF-IDF Score = 30% word + 70% char
Final Score  = 60% TF-IDF + 40% Bit-level
```

---

### 5. Classification

| Score Range | Label             |
| ----------- | ----------------- |
| ≥ 0.55      | Same Author       |
| 0.40–0.54   | Possibly Same     |
| 0.25–0.39   | Possibly Not Same |
| < 0.25      | Different Author  |

---

## 🎨 UI Features

Modern dark interface with:

* Tailwind CSS styling
* Lucide icons
* Responsive grid layout
* Dynamic result visualization
* Animated similarity bar
* Feature comparison dashboard

📌 UI implementation: 

---

## 🧪 Example Use Cases

* Detect AI-generated (deepfake) text
* Verify authorship of documents
* Compare writing styles across users
* Stylometric analysis for research

---

## 🧠 Key Concepts

* Stylometry
* N-grams
* TF-IDF Vector Space Model
* Cosine Similarity
* Feature-based NLP

---

## ⚠️ Limitations

* Not a perfect AI detector
* Sensitive to text length
* Works best with ≥ 3 samples per author
* Topic similarity can affect results

---

## 💡 Future Improvements

* Add semantic embeddings (optional ML upgrade)
* Improve threshold tuning dynamically
* Add multi-author classification
* Visualization of feature importance

---

## 🏷️ Project Title (Academic)

**Stylometric Authorship Verification using Hybrid NLP Techniques**

---

## ✨ Tagline

> “Every writer leaves a signature — InkID finds it.”
