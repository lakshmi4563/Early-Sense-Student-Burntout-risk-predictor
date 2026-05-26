import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import sys
import joblib

# Add current dir to path
sys.path.insert(0, '.')

# Set target student number (1-indexed)
STUDENT_NUMBER = 53

# Define encoders like in ml_pipeline.py
gender_enc = LabelEncoder().fit(['Female', 'Male', 'Non-binary', 'Other'])
working_enc = LabelEncoder().fit(['Both', 'Full Time', 'Not Working', 'Part Time', 'Student', 'Working'])
content_enc = LabelEncoder().fit(['Educational', 'Entertainment', 'Gaming', 'Lifestyle', 'News', 'Social'])

# Load from ml_pipeline to get the processed data
from ml_pipeline import load_test_data, FEATURE_COLS, TEST_XLSX

X_test, df_raw_test = load_test_data(TEST_XLSX, gender_enc, working_enc, content_enc)

print(f"=== STUDENT {STUDENT_NUMBER} ANALYSIS ===")
print(f"Total students: {X_test.shape[0]}")

if X_test.shape[0] >= STUDENT_NUMBER:
    student_idx = STUDENT_NUMBER - 1  # 0-indexed
    student_raw = df_raw_test.iloc[student_idx]
    student_features = X_test.iloc[student_idx].values if hasattr(X_test, 'iloc') else X_test[student_idx]

    print(f"\nCURRENT VALUES (Standardized form - what model sees):")
    for i, feat in enumerate(FEATURE_COLS):
        print(f"  {i+1:2d}. {feat:40s} = {student_features[i]:7.2f}")

    print(f"\nCURRENT VALUES (Raw form inputs - what you fill):")
    raw_fields = ['Study_Work_Hours_per_Day', 'Motivation_Level', 'Daily_Sleep_Hours', 'Exercise_Frequency_per_Week',
                  'Caffeine_Intake_Cups', 'Screen_Time_Hours', 'Night_Scrolling_Frequency', 'Online_Gaming_Hours',
                  'Overthinking_Score', 'Social_Comparison_Index', 'Anxiety_Score', 'Mood_Stability_Score',
                  'Emotional_Fatigue_Score', 'Daily_Social_Media_Hours', 'Sleep_Quality_Score', 'Age',
                  'Gender_raw', 'Working_Status_raw', 'Content_Type_raw']

    for field in raw_fields:
        if field in student_raw.index:
            print(f"  {field:40s} = {student_raw[field]}")

    # Load trained model and make prediction
    model_path = 'saved_models/burnout_model.pkl'
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        try:
            models = joblib.load(model_path)
            print(f"Model keys: {models.keys()}")
            lr = models.get('primary_model') or models.get('lr_model')
            scaler = models.get('scaler')

            if lr and scaler:
                # Get prediction
                X_scaled = scaler.transform(student_features.reshape(1, -1))
                pred = lr.predict(X_scaled)[0]
                proba = lr.predict_proba(X_scaled)[0]

                risk_labels = ['Low', 'Medium', 'High']
                print(f"\nCURRENT PREDICTION:")
                print(f"  Risk class: {risk_labels[int(pred)]}")
                print(f"  P(Low)    = {proba[0]:.4f}")
                print(f"  P(Medium) = {proba[1]:.4f}")
                print(f"  P(High)   = {proba[2]:.4f}")

                # Recommend LOW RISK values (from model_insights guide)
                print(f"\n=== TO CHANGE TO LOW RISK ===")
                print(f"Target values for LOW RISK (per model_insights guide):")
                recommendations = {
                    'Anxiety_Score': (1, 3),
                    'Motivation_Level': (8, 10),
                    'Daily_Sleep_Hours': (7, 9),
                    'Sleep_Quality_Score': (7, 10),
                    'Mood_Stability_Score': (8, 10),
                    'Emotional_Fatigue_Score': (1, 3),
                    'Overthinking_Score': (1, 3),
                    'Exercise_Frequency_per_Week': (4, 7),
                    'Study_Work_Hours_per_Day': (4, 7),
                    'Daily_Social_Media_Hours': (0, 2),
                    'Night_Scrolling_Frequency': (0, 2),
                }

                for field, (low, high) in recommendations.items():
                    current = student_raw[field]
                    print(f"  {field:40s}: {current:5} -> {low}-{high}")
            else:
                print("ERROR: Could not find lr_model or scaler in loaded model")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Model file not found: {model_path}")
else:
    print(f"Only {X_test.shape[0]} students found")
