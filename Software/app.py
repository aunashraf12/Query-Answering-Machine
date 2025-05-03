from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.utils.extmath import softmax 

app = Flask(__name__)

# Load your pre-fitted transformers & model
scaler_non_norm = joblib.load("models/scaler_non_normal.pkl")   # RobustScaler fit on a DataFrame
scaler_diff     = joblib.load("models/scaler_norm_all.pkl")    # StandardScaler fit on a DataFrame
scaler_effort   = joblib.load("models/scaler_norm_all.pkl")        # StandardScaler fit on numpy array
pca             = joblib.load("models/pca.pkl")                  # PCA(n_components=12)
model           = joblib.load("models/best_model.pkl")

# --- Config ---
NUMERIC_FEATS = [
    'course_rating','assignments_per_week','attendance_required',
    'sentiment_score','units','hours_per_week','projects',
    'midterms_count','final_exam','grading_strictness','failure_rate',
]
EFFORT_COMPONENTS = [
    'units','hours_per_week','assignments_per_week',
    'projects','midterms_count','final_exam'
]
WEIGHTS = {'units':0.4,'hours_per_week':0.3,'assignments_per_week':0.2,
           'projects':0.5,'midterms_count':0.6,'final_exam':0.4}

def assign_difficulty(row):
    fw = np.clip((row['student_percentage_estimate'] - 50)/50, 0, 1)
    wm = 1.3 if row['subject_area']=='STEM' else 1.0
    score = 0
    score += wm * row['assignments_per_week'] * 1.5
    score += wm * row['attendance_required'] * 1.0
    score -= wm * row['course_rating'] * 2 * fw
    score -= wm * row['professor_rating'] * 1.2 * fw
    score -= wm * row['sentiment_score'] * 5 * fw
    score += wm * row['grading_strictness'] * 1.0
    score += wm * row['hours_per_week'] * 0.3
    score += wm * row['midterms_count'] * 1.2
    score += wm * row['final_exam'] * 1.0
    score += wm * row['projects'] * 0.8
    score += wm * (100 - row['student_percentage_estimate']) * 0.05
    score += wm * (row['units'] - row['hours_per_week']) * -0.3
    score += wm * row['units'] * 0.3
    score += wm * row['drop_rate'] * 10
    score += wm * row['failure_rate'] * 8
    score += np.random.normal(0, 1)
    return score

def preprocess(form):

    # fields that come in as "Yes"/"No"
    bool_fields = {'attendance_required', 'final_exam'}

    # how you want to encode subject_area
    subject_map = {'STEM': 0.0, 'Humanity': 1.0, 'Social Science': 2.0}

    # build row dict by hand
    row = {}
    for feat in NUMERIC_FEATS:
        if feat in bool_fields:
            # map "Yes" -> 1.0, "No" -> 0.0
            row[feat] = 1.0 if form[feat] == 'Yes' else 0.0

        else:
            # every other field really is numeric
            row[feat] = float(form[feat])

    row['drop_rate']    = float(form['drop_rate'])    / 100.0
    row['failure_rate'] = float(form['failure_rate']) / 100.0

    # 2) Effort score
    raw_eff = sum(row[c]*WEIGHTS[c] for c in EFFORT_COMPONENTS)
    scaled_eff = scaler_effort.transform([[raw_eff]])[0,0]
    eff_clipped = np.clip(scaled_eff, -2, 2)

    # 3) Numeric features
    df_nums = pd.DataFrame([{f: row[f] for f in NUMERIC_FEATS}])
    nums_scaled = scaler_non_norm.transform(df_nums)  # (1,11)

    # 4) One-hot subject_area
    sa = form['subject_area']
    dummies = np.array([[True if sa=='Humanities' else False,
                         True if sa=='STEM'         else False,
                         True if sa=='Social Science' else False]])  # (1,3)

    # 5) Combine into 15-d vector
    feats_15 = np.hstack([nums_scaled, dummies, [[eff_clipped]]])  # (1,15)

    # 6) PCA → (1,12)
    return pca.transform(feats_15)

@app.route("/", methods=["GET","POST"])
def index():
    result = None
    if request.method=="POST":
        X12 = preprocess(request.form)

        # Try predict_proba; otherwise fallback to decision_function    
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X12)[0]
        else:
            df_vals = model.decision_function(X12)
            # binary SVC: df_vals shape (n_samples,), convert via sigmoid
            if df_vals.ndim==1:
                p1 = 1/(1+np.exp(-df_vals))
                proba = np.vstack([1-p1, p1]).T[0]
            else:
                # multiclass: apply softmax on row
                proba = softmax(df_vals)[0]


        LABEL_MAP = {
            0: "Easy",
            1: "Medium",
            2: "Hard"
        }
        raw_pred = model.classes_[np.argmax(proba)]
        label    = LABEL_MAP.get(raw_pred, str(raw_pred))

        conf     = proba[np.argmax(proba)]
        result = {"label": label, "confidence": f"{conf:.2f}"}

    return render_template("index.html", result=result)

if __name__=="__main__":
    app.run(debug=True)


