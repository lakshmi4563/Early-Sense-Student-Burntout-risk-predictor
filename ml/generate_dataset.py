"""
generate_dataset.py
Generates a synthetic training dataset for Student Burnout Risk Prediction.
Features mirror the real Google Form survey structure.
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

np.random.seed(42)
N = 1000  # synthetic samples

def generate_synthetic_data(n=N):
    data = {}

    # 1. Study hours per day (1–12)
    data['study_hours_per_day'] = np.clip(np.random.normal(6, 2.5, n), 1, 12).round(1)

    # 2. Sleep hours per day (3–10)
    data['sleep_hours_per_day'] = np.clip(np.random.normal(6.5, 1.5, n), 3, 10).round(1)

    # 3. Physical activity (hours/week) (0–10)
    data['physical_activity_hours_week'] = np.clip(np.random.exponential(2, n), 0, 10).round(1)

    # 4. Number of assignments/tasks pending (0–20)
    data['assignments_pending'] = np.clip(np.random.poisson(5, n), 0, 20).astype(int)

    # 5. Social interaction score (1–10)
    data['social_interaction_score'] = np.clip(np.random.normal(5, 2, n), 1, 10).round(1)

    # 6. Stress level self-report (1–10)
    data['stress_level'] = np.clip(np.random.normal(5.5, 2, n), 1, 10).round(1)

    # 7. Academic pressure score (1–10)
    data['academic_pressure'] = np.clip(np.random.normal(6, 2, n), 1, 10).round(1)

    # 8. Extracurricular hours per week (0–15)
    data['extracurricular_hours'] = np.clip(np.random.exponential(3, n), 0, 15).round(1)

    # 9. Motivation level (1–10)
    data['motivation_level'] = np.clip(np.random.normal(5.5, 2, n), 1, 10).round(1)

    # 10. Mental health score (1–10, higher = better)
    data['mental_health_score'] = np.clip(np.random.normal(5.5, 2, n), 1, 10).round(1)

    # 11. Attendance percentage (40–100)
    data['attendance_percentage'] = np.clip(np.random.normal(78, 15, n), 40, 100).round(1)

    # 12. Screen time non-study (hours/day) (0–8)
    data['screen_time_non_study'] = np.clip(np.random.normal(3, 1.5, n), 0, 8).round(1)

    # 13. Meal regularity (1=never, 2=sometimes, 3=always)
    data['meal_regularity'] = np.random.choice([1, 2, 3], n, p=[0.2, 0.4, 0.4])

    # 14. Peer support (1–10)
    data['peer_support'] = np.clip(np.random.normal(5.5, 2, n), 1, 10).round(1)

    # 15. Financial stress (1–10)
    data['financial_stress'] = np.clip(np.random.normal(4, 2.5, n), 1, 10).round(1)

    df = pd.DataFrame(data)

    # ---- BURN