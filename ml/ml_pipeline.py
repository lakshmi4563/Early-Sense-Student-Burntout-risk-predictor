"""
=============================================================================
Student Burnout Risk Prediction System
ML Pipeline — Training, Evaluation, and SHAP Explainability
=============================================================================

DATASETS:
  - Training  : GenZ Mental Wellness Synthetic Dataset (10,000 rows, 22 cols)
  - Testing   : Real Google Form Responses (dynamic count, 23 cols)

MODEL:
  - Primary   : Random Forest Classifier
  - Secondary : Logistic Regression (comparison)

EXPLAINABILITY:
  - Manual LinearSHAP implementation using Logistic Regression coefficients
    (Mirrors the exact algorithm of the official SHAP library's LinearExplainer)

IMBALANCE HANDLING:
  - class_weight='balanced' in RF/LR (no SMOTE dep needed)

=============================================================================
"""

import os
import re
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

# Load .env if present (for TRAIN_CSV / TEST_XLSX path overrides)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
except ImportError:
    pass

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT    = os.path.dirname(BASE_DIR)
PLOTS_DIR  = os.path.join(PROJECT, 'frontend', 'static', 'img')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset paths: set TRAIN_CSV / TEST_XLSX in .env, or place files next to this script
_DEFAULT_TRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               'genz_mental_wellness_synthetic_dataset.csv')
_DEFAULT_TEST  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'Student Wellness Survey (Responses).xlsx')
TRAIN_CSV = os.environ.get('TRAIN_CSV', _DEFAULT_TRAIN)
TEST_XLSX  = os.environ.get('TEST_XLSX', _DEFAULT_TEST)

# ─────────────────────────────────────────────
# FEATURE COLUMNS (shared between train/test)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'Study_Work_Hours_per_Day',
    'Daily_Sleep_Hours',
    'Exercise_Frequency_per_Week',
    'Caffeine_Intake_Cups',
    'Screen_Time_Hours',
    'Night_Scrolling_Frequency',
    'Online_Gaming_Hours',
    'Overthinking_Score',
    'Social_Comparison_Index',
    'Anxiety_Score',
    'Mood_Stability_Score',
    'Emotional_Fatigue_Score',
    'Daily_Social_Media_Hours',
    'Motivation_Level',
    'Sleep_Quality_Score',
    'Age',
    'Gender_enc',
    'Working_Status_enc',
    'Content_Type_enc',
]

LABEL_COL  = 'Burnout_Risk'
LABEL_MAP  = {'Low': 0, 'Medium': 1, 'High': 2}
LABEL_INV  = {0: 'Low', 1: 'Medium', 2: 'High'}

# Human-readable feature names for charts / SHAP
FEATURE_LABELS = {
    'Study_Work_Hours_per_Day':    'Study/Work Hours/Day',
    'Daily_Sleep_Hours':           'Sleep Hours/Day',
    'Exercise_Frequency_per_Week': 'Exercise Days/Week',
    'Caffeine_Intake_Cups':        'Caffeine Cups/Day',
    'Screen_Time_Hours':           'Total Screen Time (hrs)',
    'Night_Scrolling_Frequency':   'Night Scrolling Freq',
    'Online_Gaming_Hours':         'Gaming Hours/Day',
    'Overthinking_Score':          'Overthinking Score',
    'Social_Comparison_Index':     'Social Comparison',
    'Anxiety_Score':               'Anxiety Score',
    'Mood_Stability_Score':        'Mood Stability',
    'Emotional_Fatigue_Score':     'Emotional Fatigue',
    'Daily_Social_Media_Hours':    'Social Media Hrs/Day',
    'Motivation_Level':            'Motivation Level',
    'Sleep_Quality_Score':         'Sleep Quality',
    'Age':                         'Age',
    'Gender_enc':                  'Gender',
    'Working_Status_enc':          'Working Status',
    'Content_Type_enc':            'Content Type',
}


# =============================================================================
# SECTION 1: PARSE MESSY NUMERIC VALUES (from real Google Form responses)
# =============================================================================
def parse_numeric(val):
    """
    Handle entries like: '5-6', '2-3 hours', '0-2', 'No', '30.0' (gaming=30?)
    Strategy:
      - If numeric -> return float
      - If range 'a-b' or 'atob' -> return midpoint
      - If 'No' or empty -> return 0
      - Else -> return NaN (will be median-imputed)
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    if val in ('no', 'none', ''):
        return 0.0
    # Try direct numeric
    try:
        return float(val)
    except ValueError:
        pass
    # Range patterns: '5-6', '2-3 hours', '5to6', '5 to 6', '5-6 hours'
    range_match = re.search(r'(\d+\.?\d*)\s*(?:-|to)\s*(\d+\.?\d*)', val)
    if range_match:
        a, b = float(range_match.group(1)), float(range_match.group(2))
        return (a + b) / 2.0
    # Single number in string
    num_match = re.search(r'(\d+\.?\d*)', val)
    if num_match:
        return float(num_match.group(1))
    return np.nan


# =============================================================================
# SECTION 2: LOAD & CLEAN SYNTHETIC TRAINING DATA
# =============================================================================
def load_train_data(csv_path: str) -> tuple:
    """
    Load 10,000-row synthetic dataset, encode categoricals, return X, y.
    """
    df = pd.read_csv(csv_path)
    print(f"[TRAIN] Raw shape: {df.shape}")
    print(f"[TRAIN] Burnout_Risk distribution:\n{df['Burnout_Risk'].value_counts()}\n")

    # Encode categoricals — must cover ALL values in both train AND test sets
    gender_enc   = LabelEncoder().fit(['Female', 'Male', 'Non-binary', 'Other'])
    working_enc  = LabelEncoder().fit(['Both', 'Full Time', 'Not Working', 'Part Time', 'Student', 'Working'])
    content_enc  = LabelEncoder().fit(['Educational', 'Entertainment', 'Gaming', 'Lifestyle', 'News', 'Social'])

    df['Gender_enc']        = gender_enc.transform(df['Gender'].fillna('Male'))
    df['Working_Status_enc']= working_enc.transform(df['Student_Working_Status'].fillna('Student'))
    df['Content_Type_enc']  = content_enc.transform(df['Content_Type_Preference'].fillna('Entertainment'))

    # Map feature names from synthetic dataset to our unified FEATURE_COLS schema
    rename_map = {
        'Study_Work_Hours_per_Day':    'Study_Work_Hours_per_Day',
        'Daily_Sleep_Hours':           'Daily_Sleep_Hours',
        'Exercise_Frequency_per_Week': 'Exercise_Frequency_per_Week',
        'Caffeine_Intake_Cups':        'Caffeine_Intake_Cups',
        'Screen_Time_Hours':           'Screen_Time_Hours',
        'Night_Scrolling_Frequency':   'Night_Scrolling_Frequency',
        'Online_Gaming_Hours':         'Online_Gaming_Hours',
        'Overthinking_Score':          'Overthinking_Score',
        'Social_Comparison_Index':     'Social_Comparison_Index',
        'Anxiety_Score':               'Anxiety_Score',
        'Mood_Stability_Score':        'Mood_Stability_Score',
        'Emotional_Fatigue_Score':     'Emotional_Fatigue_Score',
        'Daily_Social_Media_Hours':    'Daily_Social_Media_Hours',
        'Motivation_Level':            'Motivation_Level',
        'Sleep_Quality_Score':         'Sleep_Quality_Score',
        'Age':                         'Age',
    }
    for src, dst in rename_map.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors='coerce')

    X = df[FEATURE_COLS].copy()
    # Impute any residual NaN with column median
    for col in FEATURE_COLS:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    y = df[LABEL_COL].map(LABEL_MAP)

    print(f"[TRAIN] Processed X shape: {X.shape}, missing: {X.isna().sum().sum()}")
    return X, y, gender_enc, working_enc, content_enc


# =============================================================================
# SECTION 3: LOAD & CLEAN REAL GOOGLE FORM TEST DATA
# =============================================================================
def load_test_data(xlsx_path: str, gender_enc, working_enc, content_enc) -> tuple:
    """
    Load real Google Form data (dynamic count). Heavy cleaning needed:
      - Messy numeric strings (ranges, 'No', text)
      - Outliers (sleep=16, gaming=30 -> clip to valid range)
      - Column name mapping from long question text -> schema names
    """
    df = pd.read_excel(xlsx_path)
    print(f"[TEST]  Raw shape: {df.shape}")

    # ── Column rename (keyword-based, robust to form column order changes) ──
    norm_cols = {c: re.sub(r'[^a-z0-9]+', ' ', str(c).lower()).strip() for c in df.columns}

    def find_col(required_tokens, any_tokens=None):
        any_tokens = any_tokens or []
        for original, normalized in norm_cols.items():
            if all(tok in normalized for tok in required_tokens) and (
                not any_tokens or any(tok in normalized for tok in any_tokens)
            ):
                return original
        return None

    mapping_spec = {
        'Study_Work_Hours_per_Day': (['study', 'work', 'hours'], []),
        'Working_Status_raw': (['working', 'studies'], ['status', 'alongside']),
        'Motivation_Level': (['motivation'], ['rate', 'level']),
        'Daily_Sleep_Hours': (['sleep', 'hours'], ['per day']),
        'Exercise_Frequency_per_Week': (['days', 'week', 'exercise'], []),
        'Caffeine_Intake_Cups': (['cups', 'caffeine'], ['tea', 'coffee']),
        'Screen_Time_Hours': (['screen', 'hours'], ['phone', 'laptop']),
        'Night_Scrolling_Frequency': (['scroll', 'night'], ['sleeping', 'before']),
        'Online_Gaming_Hours': (['online', 'gaming', 'hours'], []),
        'Overthinking_Score': (['overthinking'], ['rate', 'level']),
        'Social_Comparison_Index': (['compare', 'others'], []),
        'Anxiety_Score': (['anxiety'], ['rate', 'level']),
        'Mood_Stability_Score': (['mood', 'stability'], []),
        'Emotional_Fatigue_Score': (['emotional', 'fatigue'], []),
        'Daily_Social_Media_Hours': (['social', 'media', 'hours'], []),
        'Content_Type_raw': (['type', 'content'], ['look', 'online']),
        'Gender_raw': (['gender'], []),
        'Age': (['age'], ['number']),
        'Sleep_Quality_Score': (['sleep', 'quality'], ['score']),
    }

    col_rename = {}
    for target_col, (required, optional_any) in mapping_spec.items():
        found = find_col(required, optional_any)
        if found:
            col_rename[found] = target_col

    # Fallback to historical positional mapping for any fields not matched by keywords.
    legacy_positions = {
        'Study_Work_Hours_per_Day': 1,
        'Working_Status_raw': 2,
        'Motivation_Level': 3,
        'Daily_Sleep_Hours': 5,
        'Exercise_Frequency_per_Week': 6,
        'Caffeine_Intake_Cups': 7,
        'Screen_Time_Hours': 8,
        'Night_Scrolling_Frequency': 9,
        'Online_Gaming_Hours': 10,
        'Overthinking_Score': 11,
        'Social_Comparison_Index': 12,
        'Anxiety_Score': 13,
        'Mood_Stability_Score': 14,
        'Emotional_Fatigue_Score': 15,
        'Daily_Social_Media_Hours': 16,
        'Content_Type_raw': 17,
        'Gender_raw': 18,
        'Age': 19,
        'Sleep_Quality_Score': 20,
    }
    already_mapped_targets = set(col_rename.values())
    for target_col, idx in legacy_positions.items():
        if target_col in already_mapped_targets:
            continue
        if idx < len(df.columns):
            src_col = df.columns[idx]
            if src_col not in col_rename:
                col_rename[src_col] = target_col
    df.rename(columns=col_rename, inplace=True)

    # ── Numeric cleaning ──
    numeric_cols = [
        'Study_Work_Hours_per_Day', 'Daily_Sleep_Hours', 'Exercise_Frequency_per_Week',
        'Caffeine_Intake_Cups', 'Screen_Time_Hours', 'Night_Scrolling_Frequency',
        'Online_Gaming_Hours', 'Overthinking_Score', 'Social_Comparison_Index',
        'Anxiety_Score', 'Mood_Stability_Score', 'Emotional_Fatigue_Score',
        'Daily_Social_Media_Hours', 'Motivation_Level',
        'Sleep_Quality_Score', 'Age',
    ]
    for col in numeric_cols:
        df[col] = df[col].apply(parse_numeric)

    # ── Clip to valid ranges ──
    clips = {
        'Study_Work_Hours_per_Day': (0, 16),
        'Daily_Sleep_Hours':        (3, 12),   # 16 hrs sleep is impossible -> clip
        'Exercise_Frequency_per_Week': (0, 7),
        'Caffeine_Intake_Cups':     (0, 10),
        'Screen_Time_Hours':        (0, 16),
        'Night_Scrolling_Frequency':(0, 10),
        'Online_Gaming_Hours':      (0, 12),   # gaming=30 is a typo -> clip
        'Overthinking_Score':       (1, 10),
        'Social_Comparison_Index':  (1, 10),
        'Anxiety_Score':            (1, 10),
        'Mood_Stability_Score':     (1, 10),
        'Emotional_Fatigue_Score':  (1, 10),
        'Daily_Social_Media_Hours': (0, 12),
        'Motivation_Level':         (1, 10),
        'Sleep_Quality_Score':      (1, 10),
        'Age':                      (15, 35),
    }
    for col, (lo, hi) in clips.items():
        df[col] = df[col].clip(lo, hi)

    # ── Encode categoricals ──
    def safe_gender(g):
        g = str(g).strip().title()
        valid = ['Female', 'Male', 'Non-binary', 'Other']
        return g if g in valid else 'Male'

    def safe_working(w):
        w = str(w).strip().title()
        mapping = {
            'Not Working': 'Not Working',
            'Part Time': 'Part Time',
            'Full Time': 'Full Time',
            'Working': 'Working',
            'Student': 'Student',
            'Both': 'Both',
        }
        return mapping.get(w, 'Student')

    def safe_content(c):
        c = str(c).strip().title()
        valid = ['Educational', 'Entertainment', 'Gaming', 'News', 'Social']
        return c if c in valid else 'Entertainment'

    df['Gender_raw']       = df['Gender_raw'].apply(safe_gender)
    df['Working_Status_raw'] = df['Working_Status_raw'].apply(safe_working)
    df['Content_Type_raw'] = df['Content_Type_raw'].apply(safe_content)

    df['Gender_enc']        = gender_enc.transform(df['Gender_raw'])
    df['Working_Status_enc']= working_enc.transform(df['Working_Status_raw'])
    df['Content_Type_enc']  = content_enc.transform(df['Content_Type_raw'])

    X = df[FEATURE_COLS].copy()
    # Final NaN imputation with column median
    for col in FEATURE_COLS:
        med = X[col].median()
        X[col] = X[col].fillna(med if not np.isnan(med) else 0)

    print(f"[TEST]  Processed X shape: {X.shape}, missing after clean: {X.isna().sum().sum()}")
    return X, df


# =============================================================================
# SECTION 4: MANUAL LinearSHAP IMPLEMENTATION
# =============================================================================
class ManualLinearSHAP:
    """
    Implements SHAP values for Logistic Regression using the LinearSHAP method.

    Algorithm (mirrors SHAP LinearExplainer):
    ─────────────────────────────────────────
    For a Logistic Regression model with coefficient matrix W (n_classes × n_features):
      1. Compute background mean E[x] from the scaled training data.
      2. For each input sample x and predicted class c:
         phi_i = W[c, i] × (x[i] − E[x_i])
      3. This gives a per-feature contribution vector phi such that:
         Sigma phi_i = f(x) − E[f(x)]   (SHAP additivity property)

    This is the exact, closed-form solution for linear models —
    no approximation is needed. Unlike TreeSHAP (which estimates
    contributions by averaging across tree paths), LinearSHAP directly
    uses the model's own learned coefficients, making the explanation
    mathematically faithful to the prediction model itself.
    """

    def __init__(self, lr_model, feature_names: list,
                 background_mean: 'np.ndarray'):
        self.model           = lr_model
        self.feature_names   = feature_names
        self.n_features      = len(feature_names)
        self.n_classes       = len(lr_model.classes_)
        self.background_mean = background_mean   # shape (n_features,)
        # Expected value: model baseline prediction at the background mean
        self.expected_value  = lr_model.predict_proba(
            background_mean.reshape(1, -1)
        )[0]

    def shap_values(self, X: 'np.ndarray', pred_class_indices: 'np.ndarray') -> 'np.ndarray':
        """
        Compute SHAP values for a batch of samples.
        X: shape (n_samples, n_features) — scaled feature values
        pred_class_indices: shape (n_samples,) — predicted class index per sample
        Returns shap_matrix: shape (n_samples, n_features)
        """
        n_samples = X.shape[0]
        shap_matrix = np.zeros((n_samples, self.n_features))
        W = self.model.coef_   # shape (n_classes, n_features)

        for i in range(n_samples):
            cls = int(pred_class_indices[i])
            # LinearSHAP formula: phi_j = W[cls, j] × (x_j − E[x_j])
            shap_matrix[i] = W[cls] * (X[i] - self.background_mean)

        return shap_matrix

    def explain_single(self, x: 'np.ndarray', pred_class_idx: int) -> dict:
        """
        Full explanation for one sample.
        Returns dict: {feature_name: shap_value, ...} sorted by |shap_value|
        """
        W = self.model.coef_
        shap_row = W[pred_class_idx] * (x - self.background_mean)

        result = {
            self.feature_names[i]: float(shap_row[i])
            for i in range(self.n_features)
        }
        # Sort by absolute contribution (highest impact first)
        result = dict(sorted(result.items(), key=lambda kv: abs(kv[1]), reverse=True))
        return result

# =============================================================================
# SECTION 5: VISUALISATIONS
# =============================================================================
PALETTE = {'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c'}
BG      = '#0f0f23'
FG      = '#e8e8f0'

def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#1a1a35')
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    if title:   ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    if xlabel:  ax.set_xlabel(xlabel)
    if ylabel:  ax.set_ylabel(ylabel)

def plot_class_distribution(y_series: pd.Series, filename: str):
    counts = y_series.map(LABEL_INV).value_counts()
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE.get(l, '#888') for l in counts.index],
                  edgecolor='#ffffff22', linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                str(val), ha='center', va='bottom', color=FG, fontsize=10)
    _style_ax(ax, 'Burnout Risk Class Distribution (Training)', 'Risk Level', 'Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100,
                facecolor=BG)
    plt.close()
    print(f"  -> Saved {filename}")

def plot_confusion_matrix(cm: np.ndarray, model_name: str, filename: str):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=['Low','Medium','High'],
                yticklabels=['Low','Medium','High'],
                linewidths=0.5, linecolor='#222244',
                ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_facecolor('#1a1a35')
    ax.tick_params(colors=FG)
    ax.set_title(f'Confusion Matrix — {model_name}', color=FG, fontsize=12)
    ax.set_xlabel('Predicted', color=FG)
    ax.set_ylabel('Actual', color=FG)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100,
                facecolor=BG)
    plt.close()
    print(f"  -> Saved {filename}")

def plot_global_feature_importance(rf_model: RandomForestClassifier,
                                   feature_names: list, filename: str):
    """Global feature importance from RF (mean decrease impurity)."""
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1][:15]  # top 15

    labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in idx]
    values = importances[idx]
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(idx)))

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    bars = ax.barh(range(len(idx)), values[::-1], color=colors[::-1],
                   edgecolor='#ffffff11')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', color=FG, fontsize=8)
    _style_ax(ax, 'Global Feature Importance (Random Forest)',
              'Importance Score', 'Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100,
                facecolor=BG)
    plt.close()
    print(f"  -> Saved {filename}")

def plot_shap_summary(shap_matrix: np.ndarray, X_arr: np.ndarray,
                      feature_names: list, filename: str):
    """
    SHAP Summary Plot (beeswarm style).
    Each dot = one sample. X-axis = SHAP value. Color = feature value.
    """
    top_n = 12
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
    ax.set_facecolor('#1a1a35')

    jitter_y = 0.35
    for rank, feat_idx in enumerate(top_idx[::-1]):  # bottom = least important
        shap_vals  = shap_matrix[:, feat_idx]
        feat_vals  = X_arr[:, feat_idx]
        # Normalize feature values to [0,1] for color
        fmin, fmax = feat_vals.min(), feat_vals.max()
        if fmax > fmin:
            norm_feat = (feat_vals - fmin) / (fmax - fmin)
        else:
            norm_feat = np.full_like(feat_vals, 0.5)
        colors = plt.cm.RdYlBu_r(norm_feat)
        y_pos  = rank + np.random.uniform(-jitter_y, jitter_y, len(shap_vals))
        ax.scatter(shap_vals, y_pos, c=colors, s=8, alpha=0.6, linewidths=0)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=9, color=FG)
    ax.axvline(0, color='#ffffff44', linewidth=1, linestyle='--')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r',
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.set_ylabel('Feature Value (Low -> High)', color=FG, fontsize=8)
    cbar.ax.tick_params(colors=FG)

    _style_ax(ax, 'SHAP Summary Plot — Feature Impact on Burnout Prediction',
              'SHAP Value (Impact on Prediction)', '')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100,
                facecolor=BG)
    plt.close()
    print(f"  -> Saved {filename}")

def plot_shap_individual(shap_dict: dict, risk_level: str,
                         filename: str, sample_id: str = ''):
    """
    Waterfall-style SHAP plot for a single prediction.
    shap_dict: {feature_name: shap_value}  (top features already sorted)
    """
    top_n  = 10
    items  = list(shap_dict.items())[:top_n]
    # Filter only numeric, finite items
    items  = [(k, v) for k, v in items if np.isfinite(v)]
    if not items:
        return
    feats  = [FEATURE_LABELS.get(k, k) for k, _ in items]
    vals   = [v for _, v in items]
    colors = ['#e74c3c' if v > 0 else '#27ae60' for v in vals]

    n = len(feats)
    fig_h = max(4, n * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h), facecolor=BG)
    bars = ax.barh(range(n), vals[::-1], color=colors[::-1],
                   edgecolor='#ffffff11')
    ax.set_yticks(range(n))
    ax.set_yticklabels(feats[::-1], fontsize=9, color=FG)
    ax.axvline(0, color=FG, linewidth=0.8, linestyle='--')
    for bar, val in zip(bars, vals[::-1]):
        if not np.isfinite(val):
            continue
        txt = f'+{val:.4f}' if val >= 0 else f'{val:.4f}'
        xpos = bar.get_width() + (0.0005 if val >= 0 else -0.0005)
        ha   = 'left' if val >= 0 else 'right'
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                txt, va='center', ha=ha, color=FG, fontsize=8)
    risk_color = PALETTE.get(risk_level, '#888')
    title_str  = f'SHAP Explanation — Predicted: {risk_level} Burnout'
    if sample_id:
        title_str += f' (Sample {sample_id})'
    _style_ax(ax, title_str, 'SHAP Value (← Lowers Risk | Raises Risk ->)', '')
    pos_patch = mpatches.Patch(color='#e74c3c', label='↑ Increases Burnout Risk')
    neg_patch = mpatches.Patch(color='#27ae60', label='↓ Decreases Burnout Risk')
    ax.legend(handles=[pos_patch, neg_patch], facecolor='#1a1a35',
              labelcolor=FG, fontsize=8, loc='lower right')
    plt.tight_layout(pad=1.5)
    fig.set_size_inches(9, fig_h)
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100,
                facecolor=BG)
    plt.close()
    print(f"  -> Saved {filename}")

def plot_model_comparison(metrics_dict: dict, filename: str):
    """Bar chart comparing RF vs LR on multiple metrics."""
    # Filter to only numeric metrics (skip CV-F1 string)
    numeric_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    first_model_metrics = next(iter(metrics_dict.values()))
    metric_names = [m for m in numeric_metrics if m in first_model_metrics]
    
    x = np.arange(len(metric_names))
    width = 0.35
    colors = ['#6c63ff', '#ff6584']

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
        vals = [metrics[m] for m in metric_names]
        rects = ax.bar(x + idx * width, vals, width, label=model_name,
                       color=colors[idx], alpha=0.85, edgecolor='#ffffff22')
        for rect, val in zip(rects, vals):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', color=FG, fontsize=8)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(metric_names, color=FG, fontsize=9)
    ax.set_ylim(0, 1.12)
    _style_ax(ax, 'Model Performance Comparison (Test Set)', '', 'Score')
    ax.legend(facecolor='#1a1a35', labelcolor=FG, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100,
                facecolor=BG)
    plt.close()
    print(f"  OK Saved {filename}")

def plot_test_prediction_distribution(y_pred: np.ndarray, filename: str):
    """Distribution of predicted classes on the real test set."""
    labels_pred = [LABEL_INV[p] for p in y_pred]
    from collections import Counter
    counts = Counter(labels_pred)
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
    for label in ['Low', 'Medium', 'High']:
        ax.bar(label, counts.get(label, 0), color=PALETTE[label],
               edgecolor='#ffffff22')
        ax.text(label, counts.get(label, 0) + 0.3,
                str(counts.get(label, 0)), ha='center', color=FG, fontsize=11)
    _style_ax(ax, f'Predicted Burnout Risk — Real Student Data (n={len(y_pred)})',
              'Risk Level', 'Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100,
                facecolor=BG)
    plt.close()
    print(f"  -> Saved {filename}")

def plot_all_models_comparison(metrics_dict: dict, filename: str):
    """
    Comprehensive comparison of all 6 models on Accuracy, Precision, Recall, F1-Score.
    """
    models = list(metrics_dict.keys())
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Extract numeric values only (skip CV-F1 string)
    data = {model: [metrics_dict[model][m] for m in metric_cols] 
            for model in models}
    
    x = np.arange(len(metric_cols))
    width = 0.13
    colors_models = ['#6c63ff', '#00d4aa', '#ff6584', '#f39c12', '#e74c3c', '#27ae60']
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    for idx, (model, vals) in enumerate(data.items()):
        offset = (idx - len(models)/2) * width + width/2
        bars = ax.bar(x + offset, vals, width, label=model, 
                      color=colors_models[idx], alpha=0.85, edgecolor='#ffffff22')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', color=FG, fontsize=7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, color=FG, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', color=FG, fontsize=10)
    ax.set_facecolor(BG)
    ax.spines['bottom'].set_color(FG)
    ax.spines['left'].set_color(FG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=FG)
    ax.set_title('All 6 Models — Performance Comparison', color=FG, fontsize=12, fontweight='bold', pad=20)
    ax.legend(facecolor='#1a1a35', labelcolor=FG, fontsize=9, loc='lower left', ncol=2)
    ax.grid(axis='y', alpha=0.2, color=FG)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100, facecolor=BG)
    plt.close()
    print(f"  -> Saved {filename}")

def plot_grouped_shap(shap_dict: dict, feature_groups: dict, feature_labels: dict, 
                      risk_level: str, filename: str):
    """
    Plot SHAP values grouped by category (Academic, Behavioral, Lifestyle).
    Creates one subplot per group.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    group_colors = {'Academic Stress': '#6c63ff', 'Behavioural Patterns': '#00d4aa', 
                    'Lifestyle & Health': '#f39c12'}
    
    for ax_idx, (group_name, features) in enumerate(feature_groups.items()):
        ax = axes[ax_idx]
        group_shap = {f: shap_dict.get(f, 0) for f in features if f in shap_dict}
        if not group_shap:
            ax.axis('off')
            continue
        
        sorted_items = sorted(group_shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
        if not sorted_items:
            ax.axis('off')
            continue
            
        feats = [feature_labels.get(k, k) for k, _ in sorted_items]
        vals = [v for _, v in sorted_items]
        colors = ['#e74c3c' if v > 0 else '#27ae60' for v in vals]
        
        n = len(feats)
        y_pos = np.arange(n)
        ax.barh(y_pos, vals, color=colors, edgecolor='#ffffff22', alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats, fontsize=8, color=FG)
        ax.axvline(0, color=FG, linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_facecolor(BG)
        ax.spines['left'].set_color(FG)
        ax.spines['bottom'].set_color(FG)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors=FG, labelsize=7)
        ax.set_title(group_name, color=group_colors[group_name], fontsize=10, fontweight='bold')
        ax.set_xlabel('SHAP Value', color=FG, fontsize=8)
        
        for i, (feat, val) in enumerate(zip(feats, vals)):
            txt = f'{val:+.4f}'
            xpos = val + (0.001 if val >= 0 else -0.001)
            ha = 'left' if val >= 0 else 'right'
            ax.text(xpos, i, txt, va='center', ha=ha, color=FG, fontsize=7)
    
    fig.suptitle(f'SHAP Explanation (Grouped) — Predicted: {risk_level} Burnout', 
                 color=FG, fontsize=11, fontweight='bold', y=0.98)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1, wspace=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100, facecolor=BG, bbox_inches=None)
    plt.close()
    print(f"  OK Saved {filename}")

def plot_burnout_gauge(confidence_dict, risk_level: str, filename: str):
    """
    Advanced gauge/radial chart showing confidence breakdown.
    Accepts either dict {'Low': 0.1, 'Medium': 0.3, 'High': 0.6} or numpy array [0.1, 0.3, 0.6]
    """
    # Handle both dict and numpy array input
    if isinstance(confidence_dict, np.ndarray):
        values = [float(confidence_dict[0]), float(confidence_dict[1]), float(confidence_dict[2])]
    else:
        categories = ['Low', 'Medium', 'High']
        values = [float(confidence_dict.get(cat, 0)) for cat in categories]
    
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG, subplot_kw=dict(projection='polar'))
    
    categories = ['Low', 'Medium', 'High']
    colors_gauge = ['#27ae60', '#f39c12', '#e74c3c']
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]
    
    ax.plot(angles_plot, values_plot, 'o-', linewidth=2.5, color='#6c63ff', markersize=8)
    ax.fill(angles_plot, values_plot, alpha=0.25, color='#6c63ff')
    
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, color=FG, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], color=FG, fontsize=8)
    ax.grid(True, color=FG, alpha=0.2)
    
    for angle, val, cat, color in zip(angles, values, categories, colors_gauge):
        ax.text(angle, val + 0.12, f'{val:.1%}', ha='center', va='center', 
                color=color, fontsize=10, fontweight='bold')
    
    ax.set_facecolor(BG)
    title_color = PALETTE.get(risk_level, '#888')
    ax.set_title(f'Prediction Confidence — {risk_level} Risk', color=title_color, 
                fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100, facecolor=BG)
    plt.close()
    print(f"  OK Saved {filename}")


# =============================================================================
# SECTION 6: TRAIN MODELS
# =============================================================================
def train_models(X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame,   y_val: pd.Series):
    """
    Train 6 models with class_weight='balanced' to handle imbalance.
    Returns models dict, scaler, comprehensive metrics, and confusion matrices.
    """
    print("\n" + "="*60)
    print(" MODEL TRAINING (6 Models)")
    print("="*60)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_vl_scaled = scaler.transform(X_val)

    models = {}
    metrics = {}
    cms = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── 1. Random Forest ──
    print("\n[1/6] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators    = 200,
        max_depth       = 15,
        min_samples_leaf= 5,
        min_samples_split=10,
        class_weight    = 'balanced',
        n_jobs          = -1,
        random_state    = 42,
    )
    rf.fit(X_tr_scaled, y_train)
    rf_pred   = rf.predict(X_vl_scaled)
    rf_proba  = rf.predict_proba(X_vl_scaled)
    rf_acc    = accuracy_score(y_val, rf_pred)
    rf_prec   = precision_score(y_val, rf_pred, average='weighted')
    rf_rec    = recall_score(y_val, rf_pred, average='weighted')
    rf_f1     = f1_score(y_val, rf_pred, average='weighted')
    try:
        rf_auc = roc_auc_score(y_val, rf_proba, multi_class='ovr', average='weighted')
    except Exception:
        rf_auc = 0.0
    cv_rf = cross_val_score(rf, X_tr_scaled, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
    models['Random Forest'] = rf
    metrics['Random Forest'] = {
        'Accuracy': round(rf_acc, 4), 'Precision': round(rf_prec, 4), 'Recall': round(rf_rec, 4),
        'F1-Score': round(rf_f1, 4), 'AUC-ROC': round(rf_auc, 4),
        'CV-F1': f"{cv_rf.mean():.4f} ± {cv_rf.std():.4f}"
    }
    cms['Random Forest'] = confusion_matrix(y_val, rf_pred)
    print(f"  Accuracy: {rf_acc:.4f} | Precision: {rf_prec:.4f} | Recall: {rf_rec:.4f} | F1: {rf_f1:.4f} | AUC: {rf_auc:.4f}")
    print(f"  5-Fold CV F1: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

    # ── 2. Decision Tree ──
    print("\n[2/6] Training Decision Tree...")
    dt = DecisionTreeClassifier(
        max_depth       = 15,
        min_samples_leaf= 5,
        min_samples_split=10,
        class_weight    = 'balanced',
        random_state    = 42,
    )
    dt.fit(X_tr_scaled, y_train)
    dt_pred   = dt.predict(X_vl_scaled)
    dt_proba  = dt.predict_proba(X_vl_scaled)
    dt_acc    = accuracy_score(y_val, dt_pred)
    dt_prec   = precision_score(y_val, dt_pred, average='weighted')
    dt_rec    = recall_score(y_val, dt_pred, average='weighted')
    dt_f1     = f1_score(y_val, dt_pred, average='weighted')
    try:
        dt_auc = roc_auc_score(y_val, dt_proba, multi_class='ovr', average='weighted')
    except Exception:
        dt_auc = 0.0
    cv_dt = cross_val_score(dt, X_tr_scaled, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
    models['Decision Tree'] = dt
    metrics['Decision Tree'] = {
        'Accuracy': round(dt_acc, 4), 'Precision': round(dt_prec, 4), 'Recall': round(dt_rec, 4),
        'F1-Score': round(dt_f1, 4), 'AUC-ROC': round(dt_auc, 4),
        'CV-F1': f"{cv_dt.mean():.4f} ± {cv_dt.std():.4f}"
    }
    cms['Decision Tree'] = confusion_matrix(y_val, dt_pred)
    print(f"  Accuracy: {dt_acc:.4f} | Precision: {dt_prec:.4f} | Recall: {dt_rec:.4f} | F1: {dt_f1:.4f} | AUC: {dt_auc:.4f}")
    print(f"  5-Fold CV F1: {cv_dt.mean():.4f} ± {cv_dt.std():.4f}")

    # ── 3. Logistic Regression ──
    print("\n[3/6] Training Logistic Regression...")
    lr = LogisticRegression(
        C             = 1.0,
        max_iter      = 1000,
        class_weight  = 'balanced',
        solver        = 'lbfgs',
        random_state  = 42,
    )
    lr.fit(X_tr_scaled, y_train)
    lr_pred   = lr.predict(X_vl_scaled)
    lr_proba  = lr.predict_proba(X_vl_scaled)
    lr_acc    = accuracy_score(y_val, lr_pred)
    lr_prec   = precision_score(y_val, lr_pred, average='weighted')
    lr_rec    = recall_score(y_val, lr_pred, average='weighted')
    lr_f1     = f1_score(y_val, lr_pred, average='weighted')
    try:
        lr_auc = roc_auc_score(y_val, lr_proba, multi_class='ovr', average='weighted')
    except Exception:
        lr_auc = 0.0
    cv_lr = cross_val_score(lr, X_tr_scaled, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
    models['Logistic Regression'] = lr
    metrics['Logistic Regression'] = {
        'Accuracy': round(lr_acc, 4), 'Precision': round(lr_prec, 4), 'Recall': round(lr_rec, 4),
        'F1-Score': round(lr_f1, 4), 'AUC-ROC': round(lr_auc, 4),
        'CV-F1': f"{cv_lr.mean():.4f} ± {cv_lr.std():.4f}"
    }
    cms['Logistic Regression'] = confusion_matrix(y_val, lr_pred)
    print(f"  Accuracy: {lr_acc:.4f} | Precision: {lr_prec:.4f} | Recall: {lr_rec:.4f} | F1: {lr_f1:.4f} | AUC: {lr_auc:.4f}")
    print(f"  5-Fold CV F1: {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")

    # ── 4. K-Nearest Neighbors ──
    print("\n[4/6] Training K-Nearest Neighbors...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_tr_scaled, y_train)
    knn_pred  = knn.predict(X_vl_scaled)
    knn_proba = knn.predict_proba(X_vl_scaled)
    knn_acc   = accuracy_score(y_val, knn_pred)
    knn_prec  = precision_score(y_val, knn_pred, average='weighted')
    knn_rec   = recall_score(y_val, knn_pred, average='weighted')
    knn_f1    = f1_score(y_val, knn_pred, average='weighted')
    try:
        knn_auc = roc_auc_score(y_val, knn_proba, multi_class='ovr', average='weighted')
    except Exception:
        knn_auc = 0.0
    cv_knn = cross_val_score(knn, X_tr_scaled, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
    models['KNN'] = knn
    metrics['KNN'] = {
        'Accuracy': round(knn_acc, 4), 'Precision': round(knn_prec, 4), 'Recall': round(knn_rec, 4),
        'F1-Score': round(knn_f1, 4), 'AUC-ROC': round(knn_auc, 4),
        'CV-F1': f"{cv_knn.mean():.4f} ± {cv_knn.std():.4f}"
    }
    cms['KNN'] = confusion_matrix(y_val, knn_pred)
    print(f"  Accuracy: {knn_acc:.4f} | Precision: {knn_prec:.4f} | Recall: {knn_rec:.4f} | F1: {knn_f1:.4f} | AUC: {knn_auc:.4f}")
    print(f"  5-Fold CV F1: {cv_knn.mean():.4f} ± {cv_knn.std():.4f}")

    # ── 5. Support Vector Machine ──
    print("\n[5/6] Training Support Vector Machine...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', 
              probability=True, random_state=42)
    svm.fit(X_tr_scaled, y_train)
    svm_pred  = svm.predict(X_vl_scaled)
    svm_proba = svm.predict_proba(X_vl_scaled)
    svm_acc   = accuracy_score(y_val, svm_pred)
    svm_prec  = precision_score(y_val, svm_pred, average='weighted')
    svm_rec   = recall_score(y_val, svm_pred, average='weighted')
    svm_f1    = f1_score(y_val, svm_pred, average='weighted')
    try:
        svm_auc = roc_auc_score(y_val, svm_proba, multi_class='ovr', average='weighted')
    except Exception:
        svm_auc = 0.0
    cv_svm = cross_val_score(svm, X_tr_scaled, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
    models['SVM'] = svm
    metrics['SVM'] = {
        'Accuracy': round(svm_acc, 4), 'Precision': round(svm_prec, 4), 'Recall': round(svm_rec, 4),
        'F1-Score': round(svm_f1, 4), 'AUC-ROC': round(svm_auc, 4),
        'CV-F1': f"{cv_svm.mean():.4f} ± {cv_svm.std():.4f}"
    }
    cms['SVM'] = confusion_matrix(y_val, svm_pred)
    print(f"  Accuracy: {svm_acc:.4f} | Precision: {svm_prec:.4f} | Recall: {svm_rec:.4f} | F1: {svm_f1:.4f} | AUC: {svm_auc:.4f}")
    print(f"  5-Fold CV F1: {cv_svm.mean():.4f} ± {cv_svm.std():.4f}")

    # ── 6. Naive Bayes ──
    print("\n[6/6] Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_tr_scaled, y_train)
    nb_pred   = nb.predict(X_vl_scaled)
    nb_proba  = nb.predict_proba(X_vl_scaled)
    nb_acc    = accuracy_score(y_val, nb_pred)
    nb_prec   = precision_score(y_val, nb_pred, average='weighted')
    nb_rec    = recall_score(y_val, nb_pred, average='weighted')
    nb_f1     = f1_score(y_val, nb_pred, average='weighted')
    try:
        nb_auc = roc_auc_score(y_val, nb_proba, multi_class='ovr', average='weighted')
    except Exception:
        nb_auc = 0.0
    cv_nb = cross_val_score(nb, X_tr_scaled, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
    models['Naive Bayes'] = nb
    metrics['Naive Bayes'] = {
        'Accuracy': round(nb_acc, 4), 'Precision': round(nb_prec, 4), 'Recall': round(nb_rec, 4),
        'F1-Score': round(nb_f1, 4), 'AUC-ROC': round(nb_auc, 4),
        'CV-F1': f"{cv_nb.mean():.4f} ± {cv_nb.std():.4f}"
    }
    cms['Naive Bayes'] = confusion_matrix(y_val, nb_pred)
    print(f"  Accuracy: {nb_acc:.4f} | Precision: {nb_prec:.4f} | Recall: {nb_rec:.4f} | F1: {nb_f1:.4f} | AUC: {nb_auc:.4f}")
    print(f"  5-Fold CV F1: {cv_nb.mean():.4f} ± {cv_nb.std():.4f}")

    return models, scaler, metrics, cms, rf


# =============================================================================
# SECTION 7: TEST ON REAL DATA (dynamic count)
# =============================================================================
def test_on_real_data(model, scaler, X_test: pd.DataFrame, df_raw) -> dict:
    """
    Run predictions on real student responses using the production model.
    Also generate per-sample SHAP explanations for first 5 students.
    """
    print("\n" + "="*60)
    print(" TESTING ON REAL DATA")
    print("="*60)

    X_arr  = X_test.values
    X_sc   = scaler.transform(X_arr)
    y_pred = model.predict(X_sc)
    y_prob = model.predict_proba(X_sc)

    print(f"\n  Prediction distribution:")
    from collections import Counter
    dist = Counter([LABEL_INV[p] for p in y_pred])
    for k, v in dist.items():
        print(f"    {k:8s}: {v} ({100*v/len(y_pred):.1f}%)")

    # Per-student predictions
    predictions = []
    for i in range(len(y_pred)):
        predictions.append({
            'student_id': i + 1,
            'predicted_risk': LABEL_INV[y_pred[i]],
            'confidence': {
                'Low':    round(float(y_prob[i][0]), 4),
                'Medium': round(float(y_prob[i][1]), 4),
                'High':   round(float(y_prob[i][2]), 4),
            },
        })

    return {
        'predictions': predictions,
        'distribution': dict(dist),
        'X_arr': X_arr,
        'y_pred': y_pred,
        'y_prob': y_prob,
    }


# =============================================================================
# SECTION 7b: REAL-DATA CONFIDENCE METRICS (no ground truth — confidence-based)
# =============================================================================
def compute_real_data_metrics(y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Since real Google Form responses have NO ground-truth burnout labels,
    traditional accuracy / F1 cannot be computed.

    Instead we compute:
      1. Prediction distribution (% per class)
      2. Mean prediction confidence (avg max-class probability)
      3. Confidence calibration bands (90%, 99%, 100%)
      4. Per-class confidence statistics (mean/min/max)
      5. Certainty ratio — fraction of predictions the model is >90% sure about

    These are the standard deployment-phase metrics used when labels are unavailable.
    """
    from collections import Counter
    n = len(y_pred)
    labels_pred   = [LABEL_INV[int(p)] for p in y_pred]
    dist          = Counter(labels_pred)

    # Max confidence per prediction (confidence in the predicted class)
    max_confs = np.array([y_prob[i, int(y_pred[i])] for i in range(n)])

    # Per-class confidence
    per_class = {}
    for cls_idx, cls_name in LABEL_INV.items():
        probs = y_prob[:, cls_idx]
        per_class[cls_name] = {
            'mean': round(float(probs.mean()), 4),
            'min':  round(float(probs.min()),  4),
            'max':  round(float(probs.max()),  4),
            'std':  round(float(probs.std()),  4),
        }

    metrics = {
        'note': (
            'Real test data has no ground-truth labels. '
            'Metrics are computed from model confidence (probability calibration).'
        ),
        'n_real_responses': n,
        'prediction_distribution': {
            cls: {
                'count': dist.get(cls, 0),
                'percentage': round(100 * dist.get(cls, 0) / n, 1),
            }
            for cls in ['Low', 'Medium', 'High']
        },
        'confidence': {
            'mean_confidence_pct':   round(float(max_confs.mean()) * 100, 2),
            'median_confidence_pct': round(float(np.median(max_confs)) * 100, 2),
            'min_confidence_pct':    round(float(max_confs.min()) * 100, 2),
            'max_confidence_pct':    round(float(max_confs.max()) * 100, 2),
            'std_confidence_pct':    round(float(max_confs.std()) * 100, 2),
            'pct_above_90':  round(100 * np.sum(max_confs >= 0.90) / n, 1),
            'pct_above_99':  round(100 * np.sum(max_confs >= 0.99) / n, 1),
            'pct_at_100':    round(100 * np.sum(max_confs == 1.00) / n, 1),
            'n_above_90':  int(np.sum(max_confs >= 0.90)),
            'n_above_99':  int(np.sum(max_confs >= 0.99)),
            'n_at_100':    int(np.sum(max_confs == 1.00)),
        },
        'per_class_confidence': per_class,
    }

    # Print summary
    print("\n" + "="*60)
    print(" REAL DATA EVALUATION METRICS (Confidence-Based)")
    print("="*60)
    print(f"  Total real responses evaluated : {n}")
    print(f"  Prediction distribution:")
    for cls in ['Low', 'Medium', 'High']:
        c = dist.get(cls, 0)
        print(f"    {cls:8s}: {c:3d} ({100*c/n:.1f}%)")
    print(f"\n  Model Confidence (on predicted class):")
    print(f"    Mean      : {metrics['confidence']['mean_confidence_pct']:.2f}%")
    print(f"    Median    : {metrics['confidence']['median_confidence_pct']:.2f}%")
    print(f"    Min       : {metrics['confidence']['min_confidence_pct']:.2f}%")
    print(f"    Max       : {metrics['confidence']['max_confidence_pct']:.2f}%")
    print(f"    >= 90%    : {metrics['confidence']['n_above_90']}/{n}  ({metrics['confidence']['pct_above_90']}%)")
    print(f"    >= 99%    : {metrics['confidence']['n_above_99']}/{n}  ({metrics['confidence']['pct_above_99']}%)")
    print(f"    = 100%    : {metrics['confidence']['n_at_100']}/{n}  ({metrics['confidence']['pct_at_100']}%)")
    return metrics


def plot_real_data_confidence_chart(y_pred: np.ndarray, y_prob: np.ndarray,
                                    real_metrics: dict, filename: str):
    """
    2-panel chart:
      Left  — Prediction distribution (bar chart)
      Right — Confidence histogram (how sure the model is per prediction)
    """
    from collections import Counter
    labels_pred = [LABEL_INV[int(p)] for p in y_pred]
    dist        = Counter(labels_pred)
    max_confs   = np.array([y_prob[i, int(y_pred[i])] for i in range(len(y_pred))])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)

    # ── Left: Distribution ──
    ax1.set_facecolor('#1a1a35')
    for cls in ['Low', 'Medium', 'High']:
        cnt = dist.get(cls, 0)
        bar = ax1.bar(cls, cnt, color=PALETTE[cls], edgecolor='#ffffff22')
        ax1.text(cls, cnt + 0.3, f'{cnt}\n({100*cnt/len(y_pred):.1f}%)',
                 ha='center', color=FG, fontsize=9, fontweight='bold')
    _style_ax(ax1,
              f'Predicted Risk Distribution — Real Data (n={len(y_pred)})',
              'Risk Level', 'Number of Students')

    # ── Right: Confidence histogram ──
    ax2.set_facecolor('#1a1a35')
    colors_conf = ['#e74c3c' if c < 0.80 else '#f39c12' if c < 0.90 else '#27ae60'
                   for c in max_confs]
    ax2.bar(range(len(max_confs)), sorted(max_confs),
            color=sorted(colors_conf), edgecolor='#ffffff11', width=0.9)
    ax2.axhline(0.90, color='#f39c12', linewidth=1.2, linestyle='--',
                label='90% threshold')
    ax2.axhline(0.99, color='#27ae60', linewidth=1.2, linestyle='--',
                label='99% threshold')
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0])
    ax2.set_yticklabels(['0%','25%','50%','75%','90%','99%','100%'],
                        color=FG, fontsize=8)
    ax2.set_xlabel('Student (sorted by confidence)', color=FG)
    ax2.set_ylabel('Prediction Confidence', color=FG)
    mean_c = real_metrics['confidence']['mean_confidence_pct']
    ax2.set_title(f'Model Confidence per Student  [Mean: {mean_c:.1f}%]',
                  color=FG, fontsize=11, fontweight='bold', pad=10)
    for spine in ax2.spines.values(): spine.set_edgecolor('#333355')
    ax2.tick_params(colors=FG)
    legend = ax2.legend(facecolor='#1a1a35', labelcolor=FG, fontsize=8)

    # Confidence band annotations
    n = len(max_confs)
    n90 = real_metrics['confidence']['n_above_90']
    ax2.text(n * 0.02, 0.92,
             f"{n90}/{n} predictions ≥ 90% confident",
             color='#f39c12', fontsize=8)

    plt.suptitle('Real Student Data — Deployment Evaluation',
                 color=FG, fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100,
                facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {filename}")


# =============================================================================
# SECTION 8: SAVE ARTEFACTS
# =============================================================================
def save_artefacts(lr, scaler, feature_cols, label_map, label_inv,
                   gender_enc, working_enc, content_enc, background_mean,
                   all_models=None):
    """Pickle all model artefacts needed by the Flask backend."""
    # Primary model: Logistic Regression (99.40% accuracy, 99.99% AUC-ROC)
    # SHAP: ManualLinearSHAP using LR coefficients + background_mean
    bundle = {
        'primary_model': lr,       # Logistic Regression — predictions & SHAP
        'lr_model':      lr,       # Explicit reference
        'scaler':        scaler,
        'feature_cols':  feature_cols,
        'label_map':     label_map,
        'label_inv':     label_inv,
        'gender_enc':    gender_enc,
        'working_enc':   working_enc,
        'content_enc':   content_enc,
        'feature_labels': FEATURE_LABELS,
        'background_mean': background_mean,  # Required for ManualLinearSHAP
    }
    # Add all trained models if provided
    if all_models:
        bundle['all_models'] = all_models
    path = os.path.join(MODELS_DIR, 'burnout_model.pkl')
    with open(path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"\n  OK Model bundle saved -> {path}")
    return path


# =============================================================================
# MAIN: Full Training + Evaluation Pipeline
# =============================================================================
def run_full_pipeline():
    print("\n" + "="*60)
    print("  STUDENT BURNOUT RISK PREDICTION — ML PIPELINE")
    print("  RV University — Nandini — BTech CSE Final Year")
    print("="*60 + "\n")

    # ── 1. Load training data ──
    X_all, y_all, gender_enc, working_enc, content_enc = load_train_data(TRAIN_CSV)
    plot_class_distribution(y_all, 'class_distribution.png')

    # ── 2. Train/val split ──
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(f"\n[SPLIT] Train: {X_train.shape}, Val: {X_val.shape}")

    # ── 3. Train models ──
    models, scaler, metrics, cms, rf = train_models(
        X_train, y_train, X_val, y_val
    )
    lr = models['Logistic Regression']

    # ── 4. Confusion matrices (RF and LR only) ──
    print("\n[VIZ] Generating confusion matrices...")
    plot_confusion_matrix(cms['Random Forest'], 'Random Forest', 'cm_rf.png')
    plot_confusion_matrix(cms['Logistic Regression'], 'Logistic Regression', 'cm_lr.png')

    # ── 5. Global feature importance ──
    print("[VIZ] Global feature importance...")
    plot_global_feature_importance(rf, FEATURE_COLS, 'feature_importance.png')

    # ── 6. Model comparison charts ──
    print("[VIZ] Model comparison...")
    plot_model_comparison({k: v for k, v in list(metrics.items())[:2]}, 'model_comparison.png')
    plot_all_models_comparison(metrics, 'model_comparison_all.png')

    # ── 7. SHAP values on validation set ──
    print("\n[SHAP] Computing SHAP values on validation set (200 samples)...")
    X_val_arr    = X_val.values
    X_val_scaled = scaler.transform(X_val_arr)
    y_val_pred   = lr.predict(X_val_scaled)

    # Compute background mean for LinearSHAP (mean of scaled training data)
    background_mean = scaler.transform(X_train.values).mean(axis=0)
    shap_explainer = ManualLinearSHAP(lr, FEATURE_COLS, background_mean)
    # Compute on a 200-sample subset for speed
    n_shap = min(200, len(X_val_arr))
    shap_mat = shap_explainer.shap_values(
        X_val_scaled[:n_shap], y_val_pred[:n_shap]
    )
    print(f"  SHAP matrix shape: {shap_mat.shape}")

    print("[VIZ] SHAP summary plot...")
    plot_shap_summary(shap_mat, X_val_scaled[:n_shap], FEATURE_COLS, 'shap_summary.png')

    # ── 8. Individual SHAP for 3 validation examples ──
    print("[VIZ] Individual SHAP waterfall plots...")
    for sample_idx in [0, 1, 2]:
        shap_dict = shap_explainer.explain_single(
            X_val_scaled[sample_idx], int(y_val_pred[sample_idx])
        )
        risk_label = LABEL_INV[int(y_val_pred[sample_idx])]
        plot_shap_individual(shap_dict, risk_label,
                             f'shap_individual_{sample_idx}.png',
                             sample_id=str(sample_idx+1))

    # ── 9. Test on real response data ──
    X_test, df_raw_test = load_test_data(TEST_XLSX, gender_enc, working_enc, content_enc)
    test_results = test_on_real_data(lr, scaler, X_test, df_raw_test)
    plot_test_prediction_distribution(test_results['y_pred'], 'test_predictions.png')

    # ── 10. SHAP on all real samples ──
    print("\n[SHAP] SHAP on real test data...")
    X_test_arr    = test_results['X_arr']
    X_test_scaled = scaler.transform(X_test_arr)
    y_test_pred   = test_results['y_pred']
    shap_mat_test = shap_explainer.shap_values(X_test_scaled, y_test_pred)
    plot_shap_summary(shap_mat_test, X_test_scaled, FEATURE_COLS,
                      'shap_summary_real.png')

    # Generate individual SHAP for first 5 real students
    for i in range(min(5, len(X_test_scaled))):
        sd = shap_explainer.explain_single(X_test_scaled[i], int(y_test_pred[i]))
        plot_shap_individual(sd, LABEL_INV[int(y_test_pred[i])],
                             f'shap_real_student_{i+1}.png', sample_id=str(i+1))
    
    # Generate grouped SHAP for first 3 students
    print("[VIZ] Grouped SHAP plots...")
    FEATURE_GROUPS = {
        'Academic Stress': ['Study_Work_Hours_per_Day', 'Overthinking_Score', 'Anxiety_Score', 'Motivation_Level', 'Working_Status_enc'],
        'Behavioural Patterns': ['Screen_Time_Hours', 'Night_Scrolling_Frequency', 'Online_Gaming_Hours', 'Daily_Social_Media_Hours', 'Caffeine_Intake_Cups', 'Content_Type_enc'],
        'Lifestyle & Health': ['Daily_Sleep_Hours', 'Sleep_Quality_Score', 'Exercise_Frequency_per_Week', 'Mood_Stability_Score', 'Emotional_Fatigue_Score', 'Social_Comparison_Index', 'Age', 'Gender_enc'],
    }
    for i in range(min(3, len(X_test_scaled))):
        sd = shap_explainer.explain_single(X_test_scaled[i], int(y_test_pred[i]))
        risk_label = LABEL_INV[int(y_test_pred[i])]
        plot_grouped_shap(sd, FEATURE_GROUPS, FEATURE_LABELS, risk_label,
                          f'shap_grouped_student_{i+1}.png')
        # Add gauge chart for confidence breakdown
        plot_burnout_gauge(test_results['y_prob'][i], risk_label,
                          f'confidence_gauge_student_{i+1}.png')

    # ── 11. Save all artefacts ──
    save_artefacts(lr, scaler, FEATURE_COLS, LABEL_MAP, LABEL_INV,
                   gender_enc, working_enc, content_enc, background_mean,
                   all_models=models)

    # ── 12. Real-data confidence metrics ──
    print("\n[EVAL] Computing real-data confidence metrics...")
    real_metrics = compute_real_data_metrics(
        test_results['y_pred'], test_results['y_prob']
    )
    print("[VIZ] Real-data confidence chart...")
    plot_real_data_confidence_chart(
        test_results['y_pred'], test_results['y_prob'],
        real_metrics, 'real_data_confidence.png'
    )

    # ── 13. Save metrics JSON for Flask to read ──
    metrics_out = {
        'validation': metrics,
        'test_distribution': test_results['distribution'],
        'student_predictions': test_results['predictions'],
        'n_test': len(test_results['predictions']),
        'real_test_metrics': real_metrics,
    }
    metrics_path = os.path.join(MODELS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  OK Metrics saved -> {metrics_path}")

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE OK")
    print("  All models, encoders, plots, metrics saved.")
    print("="*60 + "\n")

    return lr, scaler, shap_explainer


if __name__ == '__main__':
    run_full_pipeline()
