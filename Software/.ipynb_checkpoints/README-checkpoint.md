# Course Difficulty Predictor Web App

This repository contains a Flask-based web application for predicting course difficulty based on user inputs.

---

## Features

* **Interactive form** collecting course attributes and student inputs:

  * Course Rating (1–5)
  * Assignments per Week
  * Attendance Required (Yes/No)
  * Sentiment Score (1–5)
  * Units
  * Hours per Week
  * Projects
  * Midterms Count
  * Final Exam (Yes/No)
  * Grading Strictness (1–5)
  * Failure Rate (%)
  * Student Percentage Estimate (%)
  * Drop Rate (%)
  * Subject Area (STEM, Humanity, Social Science)

* **Backend preprocessing** converts inputs:

  * Yes/No → numeric (1.0/0.0)
  * Ratings → integers 1–5
  * Percent fields normalized where appropriate
  * Subject area encoded to numeric labels

* **Scaling & PCA** using pre‑fitted transformers

* **Machine learning model** predicts difficulty label (Easy/Medium/Hard)

* **Softmax probabilities** displayed as confidence score

---

## Repository Structure

```
├── app.py             # Flask application
├── index.html         # HTML template for form and results display
├── models/            # Pre‑fitted scalers, PCA, and trained model
│   ├── scaler_non_normal.pkl
│   ├── scaler_norm_all.pkl
│   ├── pca.pkl
│   └── best_model.pkl
├── templates/
│   └── index.html     # template used by Flask
└── README.md          # this file
```

---

## Requirements

* Python 3.8+
* Flask
* joblib
* numpy
* pandas
* scikit-learn

Install via:

```bash
pip install flask joblib numpy pandas scikit-learn
```

---

## Installation & Running


1. **Verify models folder**
   Ensure `models/` contains the four `.pkl` files.

2. **Run the app**
   ```bash
    python app.py
   ````

4. Open your browser to `http://127.0.0.1:5000/`

---

## Usage

1. Fill out each field in the form. See below for accepted ranges:

   * **Course Rating, Professor Rating, Sentiment Score, Grading Strictness**: integers 1–5
   * **Attendance Required, Final Exam**: select Yes or No
   * **Percent estimates**:

     * Student Percentage Estimate: 0–100 (kept as is)
     * Failure Rate, Drop Rate: 0–100 (converted internally to 0.0–1.0)
   * **Subject Area**: choose one of STEM, Humanity, Social Science

2. Click **Predict**.

3. The page will reload showing:

   * **Difficulty** label (Easy/Medium/Hard)
   * **Confidence** score (softmax probability)

---

## Customization

* To change encoding or scaling, edit `preprocess()` in `app.py`.
* To swap out the model or PCA, replace files in `models/` and restart.

---

