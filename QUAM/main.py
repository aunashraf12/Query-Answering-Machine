import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import joblib

# Load saved model and preprocessing tools
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

# All 15 features
dropdown_features = {
    'course_rating': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    'assignments_per_week': list(range(0, 6)),
    'attendance_required': [0, 1],
    'professor_rating': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    'sentiment_score': [-1.0, -0.5, 0.0, 0.5, 1.0],
    'subject_area': ['STEM', 'Humanities', 'Social Science'],
    'units': [3, 6, 9, 12],
    'hours_per_week': list(range(1, 21)),
    'projects': [0, 1],
    'midterms_count': [0, 1, 2],
    'final_exam': [0, 1],
    'grading_strictness': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    'student_percentage_estimate': list(range(50, 101, 5)),
    'drop_rate': [round(x, 2) for x in np.linspace(0.0, 0.3, 7)],
    'failure_rate': [round(x, 2) for x in np.linspace(0.0, 0.4, 9)]
}

# GUI dictionary to store selected values
dropdown_vars = {}

# Create main window
root = tk.Tk()
root.title("QuAM - Course Difficulty Predictor (Dropdowns)")
root.geometry("500x950")

tk.Label(root, text="Select Course Features", font=("Helvetica", 14)).pack(pady=10)

# Create dropdowns
for feature, options in dropdown_features.items():
    label = tk.Label(root, text=feature)
    label.pack()
    
    var = tk.StringVar()
    dropdown = ttk.Combobox(root, textvariable=var, values=options, state="readonly")
    dropdown.current(0)
    dropdown.pack()
    
    dropdown_vars[feature] = var

# Prediction function
def predict_difficulty():
    try:
        input_data = []
        for feature in dropdown_features.keys():
            value = dropdown_vars[feature].get()

            # Convert all to float for the model
            if feature == 'subject_area':
                subject_map = {'STEM': 0, 'Humanities': 1, 'Social Science': 2}
                value = subject_map[value]
            else:
                value = float(value)

            input_data.append(value)

        # Preprocess
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        input_pca = pca.transform(input_scaled)
        prediction = model.predict(input_pca)

        # Map result
        difficulty_map = {0: "Easy", 1: "Medium", 2: "Hard"}
        result = difficulty_map[prediction[0]]

        messagebox.showinfo("Prediction", f"🔮 Predicted Difficulty Level: {result}")

    except Exception as e:
        messagebox.showerror("Error", f" Invalid input: {e}")

# Predict button
tk.Button(root, text="Predict Course Difficulty", command=predict_difficulty).pack(pady=20)

# Start GUI
root.mainloop()
