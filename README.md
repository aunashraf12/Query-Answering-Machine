
# Query Answering Machine (QuAM)

A machine learning project developed as part of 15-288 at Carnegie Mellon University Qatar. The system predicts university course difficulty using structured features and provides a user-friendly interface for interaction.

---

## Project Structure

```
Query-Answering-Machine/
│
├── Software/
│   ├── app.py                      # Main Flask web application
│   ├── templates/
│   │   └── index.html              # Frontend interface for user input
│   ├── models/
│   │   ├── best_model.pkl          # Trained RBF SVM model
│   │   ├── pca.pkl                 # PCA transformer
│   │   ├── scaler.pkl              # Main scaler for numeric features
│   │   ├── scaler_non_normal.pkl   # Robust scaler for non-normal features
│   │   └── scaler_norm_all.pkl     # Standard scaler for normalized features
│   ├── cleaned_course_difficulty_dataset.csv
│   ├── course_difficulty_dataset.csv
│   ├── dirty_course_difficulty_dataset.csv
│   ├── Project.ipynb              # Development notebook
│   ├── QuAM Basics.ipynb          # Model iterations and results
│   ├── QuAM Report.ipynb          # Final report and visualizations
│   └── README.md                  # This file
│
├── cleaned_course_difficulty_dataset.csv
├── course_difficulty_dataset.csv
├── dirty_course_difficulty_dataset.csv
├── Project.ipynb
├── QuAM Basics.ipynb
└── QuAM Report.ipynb
```

---

## How to Use the Software

1. **Setup environment**
   - Install required packages using:
     ```bash
     pip install flask scikit-learn pandas numpy
     ```

2. **Run the application**
   - Navigate to the `Software/` directory:
     ```bash
     cd Software
     python app.py
     ```
   - Open your browser and go to: [http://localhost:5000](http://localhost:5000)

3. **Enter input**
   - Use the dropdowns and inputs to enter course features.
   - The app will return the predicted difficulty class (Easy, Medium, Hard).

---

## Model Details

- **Model Used**: RBF Kernel SVM with PCA preprocessing.

---

## Notebooks Overview

- `QuAM Basics.ipynb`: All three model iterations (KNN, Decision Trees, SVM) with tuning and evaluation.
- `QuAM Report.ipynb`: Final report combining rationale, visualizations, and lessons learned.
- `Project.ipynb`: Consolidated notebook for Data Acquisition/Generation, Analysis, Wrangling, & Feature Engineering

---

## Notes

- The `models/` folder contains serialized (`.pkl`) versions of models and scalers used in deployment.
- All datasets are included in both raw (`dirty`) and cleaned forms for transparency but we train our model on `cleaned_course_difficulty_dataset.csv`.
- A `.git/` folder is included to show version control history but can be ignored for use.

---

## Authors

- Aun Muhammad Ashraf
- Anas Semsayan

CMU-Q, 15-288: Machine Learning in a Nutshell (Spring 2025)
