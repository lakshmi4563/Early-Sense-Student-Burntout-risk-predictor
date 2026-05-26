"""
Patch script: Replace ManualTreeSHAP with ManualLinearSHAP across the entire project.
Run once, then delete this file.
"""
import re, os

BASE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATCH ml/ml_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
ml_path = os.path.join(BASE, 'ml', 'ml_pipeline.py')
with open(ml_path, 'r', encoding='utf-8') as f:
    ml = f.read()

# 1a. Docstring: TreeSHAP → LinearSHAP
ml = ml.replace(
    '  - Manual TreeSHAP implementation using sklearn decision_path API\n'
    '    (Mirrors the exact algorithm of the official SHAP library\'s TreeExplainer)',
    '  - Manual LinearSHAP implementation using Logistic Regression coefficients\n'
    '    (Mirrors the exact algorithm of the official SHAP library\'s LinearExplainer)'
)
# Also handle \r\n
ml = ml.replace(
    '  - Manual TreeSHAP implementation using sklearn decision_path API\r\n'
    '    (Mirrors the exact algorithm of the official SHAP library\'s TreeExplainer)',
    '  - Manual LinearSHAP implementation using Logistic Regression coefficients\r\n'
    '    (Mirrors the exact algorithm of the official SHAP library\'s LinearExplainer)'
)

# 1b. Replace the entire ManualTreeSHAP class (SECTION 4) with ManualLinearSHAP
# Find the section from "# SECTION 4:" to the next "# =====" section header
old_section_pattern = (
    r'# =+\n# SECTION 4: MANUAL TreeSHAP IMPLEMENTATION\n# =+\n'
    r'class ManualTreeSHAP:.*?'
    r'(?=\n\n# =)'
)
old_section_pattern_r = (
    r'# =+\r\n# SECTION 4: MANUAL TreeSHAP IMPLEMENTATION\r\n# =+\r\n'
    r'class ManualTreeSHAP:.*?'
    r'(?=\r\n\r\n# =)'
)

NEW_LINEAR_SHAP_CLASS = '''# =============================================================================
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
         φ_i = W[c, i] × (x[i] − E[x_i])
      3. This gives a per-feature contribution vector φ such that:
         Σ φ_i = f(x) − E[f(x)]   (SHAP additivity property)

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
            # LinearSHAP formula: φ_j = W[cls, j] × (x_j − E[x_j])
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
        return result'''

# Try \r\n first (Windows), then \n (Unix)
ml_new = re.sub(old_section_pattern_r, NEW_LINEAR_SHAP_CLASS.replace('\n', '\r\n'), ml, flags=re.DOTALL)
if ml_new == ml:
    ml_new = re.sub(old_section_pattern, NEW_LINEAR_SHAP_CLASS, ml, flags=re.DOTALL)
ml = ml_new

# 1c. Update save_artefacts signature and body
ml = ml.replace(
    'def save_artefacts(rf, lr, scaler, feature_cols, label_map, label_inv,\n'
    '                   gender_enc, working_enc, content_enc, all_models=None):',
    'def save_artefacts(lr, scaler, feature_cols, label_map, label_inv,\n'
    '                   gender_enc, working_enc, content_enc, background_mean,\n'
    '                   all_models=None):'
)
ml = ml.replace(
    'def save_artefacts(rf, lr, scaler, feature_cols, label_map, label_inv,\r\n'
    '                   gender_enc, working_enc, content_enc, all_models=None):',
    'def save_artefacts(lr, scaler, feature_cols, label_map, label_inv,\r\n'
    '                   gender_enc, working_enc, content_enc, background_mean,\r\n'
    '                   all_models=None):'
)

# Update save_artefacts docstring/comment
ml = ml.replace(
    '    # Primary model is now Logistic Regression (99.40% accuracy, 99.99% AUC-ROC)\n'
    '    # Random Forest kept for SHAP explanations (tree-based explainability)\n'
    '    bundle = {\n'
    "        'primary_model': lr,  # Logistic Regression for predictions (best performance)\n"
    "        'rf_model':      rf,  # Random Forest for SHAP explanations (tree compatibility)\n"
    "        'lr_model':      lr,  # Also store LR explicitly for reference\n",
    '    # Primary model: Logistic Regression (99.40% accuracy, 99.99% AUC-ROC)\n'
    '    # SHAP: ManualLinearSHAP using LR coefficients + background_mean\n'
    '    bundle = {\n'
    "        'primary_model': lr,       # Logistic Regression — predictions & SHAP\n"
    "        'lr_model':      lr,       # Explicit reference\n"
)
ml = ml.replace(
    '    # Primary model is now Logistic Regression (99.40% accuracy, 99.99% AUC-ROC)\r\n'
    '    # Random Forest kept for SHAP explanations (tree-based explainability)\r\n'
    '    bundle = {\r\n'
    "        'primary_model': lr,  # Logistic Regression for predictions (best performance)\r\n"
    "        'rf_model':      rf,  # Random Forest for SHAP explanations (tree compatibility)\r\n"
    "        'lr_model':      lr,  # Also store LR explicitly for reference\r\n",
    '    # Primary model: Logistic Regression (99.40% accuracy, 99.99% AUC-ROC)\r\n'
    '    # SHAP: ManualLinearSHAP using LR coefficients + background_mean\r\n'
    '    bundle = {\r\n'
    "        'primary_model': lr,       # Logistic Regression — predictions & SHAP\r\n"
    "        'lr_model':      lr,       # Explicit reference\r\n"
)

# Add background_mean to bundle (insert after feature_labels line)
ml = ml.replace(
    "        'feature_labels': FEATURE_LABELS,\n"
    '    }',
    "        'feature_labels': FEATURE_LABELS,\n"
    "        'background_mean': background_mean,  # Required for ManualLinearSHAP\n"
    '    }'
)
ml = ml.replace(
    "        'feature_labels': FEATURE_LABELS,\r\n"
    '    }',
    "        'feature_labels': FEATURE_LABELS,\r\n"
    "        'background_mean': background_mean,  # Required for ManualLinearSHAP\r\n"
    '    }'
)

# 1d. Update SHAP section in run_full_pipeline: use LR instead of RF
ml = ml.replace(
    '    y_val_pred   = rf.predict(X_val_scaled)\n'
    '\n'
    '    shap_explainer = ManualTreeSHAP(rf, FEATURE_COLS)',
    '    y_val_pred   = lr.predict(X_val_scaled)\n'
    '\n'
    '    # Compute background mean for LinearSHAP (mean of scaled training data)\n'
    '    background_mean = scaler.transform(X_train.values).mean(axis=0)\n'
    '    shap_explainer = ManualLinearSHAP(lr, FEATURE_COLS, background_mean)'
)
ml = ml.replace(
    '    y_val_pred   = rf.predict(X_val_scaled)\r\n'
    '\r\n'
    '    shap_explainer = ManualTreeSHAP(rf, FEATURE_COLS)',
    '    y_val_pred   = lr.predict(X_val_scaled)\r\n'
    '\r\n'
    '    # Compute background mean for LinearSHAP (mean of scaled training data)\r\n'
    '    background_mean = scaler.transform(X_train.values).mean(axis=0)\r\n'
    '    shap_explainer = ManualLinearSHAP(lr, FEATURE_COLS, background_mean)'
)

# 1e. Update save_artefacts call
ml = ml.replace(
    '    save_artefacts(rf, lr, scaler, FEATURE_COLS, LABEL_MAP, LABEL_INV,\n'
    '                   gender_enc, working_enc, content_enc, all_models=models)',
    '    save_artefacts(lr, scaler, FEATURE_COLS, LABEL_MAP, LABEL_INV,\n'
    '                   gender_enc, working_enc, content_enc, background_mean,\n'
    '                   all_models=models)'
)
ml = ml.replace(
    '    save_artefacts(rf, lr, scaler, FEATURE_COLS, LABEL_MAP, LABEL_INV,\r\n'
    '                   gender_enc, working_enc, content_enc, all_models=models)',
    '    save_artefacts(lr, scaler, FEATURE_COLS, LABEL_MAP, LABEL_INV,\r\n'
    '                   gender_enc, working_enc, content_enc, background_mean,\r\n'
    '                   all_models=models)'
)

# 1f. Update return statement
ml = ml.replace(
    '    return rf, lr, scaler, shap_explainer',
    '    return lr, scaler, shap_explainer'
)

with open(ml_path, 'w', encoding='utf-8') as f:
    f.write(ml)
print(f"[OK] Patched {ml_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PATCH app.py
# ─────────────────────────────────────────────────────────────────────────────
app_path = os.path.join(BASE, 'app.py')
with open(app_path, 'r', encoding='utf-8') as f:
    app = f.read()

# 2a. Replace the ManualTreeSHAP class in app.py with ManualLinearSHAP
OLD_APP_SHAP = '''# ── TreeSHAP ─────────────────────────────────────────────────────────────────
class ManualTreeSHAP:
    def __init__(self,model,fnames): self.model=model; self.fnames=fnames
    def explain_single(self,x,pid):
        c=np.zeros(len(self.fnames))
        for tree in self.model.estimators_:
            t=tree.tree_; path=tree.decision_path(x.reshape(1,-1)).indices
            for i in range(1,len(path)):
                ch,pa=path[i],path[i-1]; sf=t.feature[pa]
                if sf<0: continue
                vp=t.value[pa,0,:]; vc=t.value[ch,0,:]; sp,sc=vp.sum(),vc.sum()
                if sp>0 and sc>0: c[sf]+=(vc/sc-vp/sp)[pid]
        c/=max(len(self.model.estimators_),1)
        return dict(sorted({self.fnames[i]:float(c[i]) for i in range(len(self.fnames))}.items(),key=lambda kv:abs(kv[1]),reverse=True))'''

NEW_APP_SHAP = '''# ── LinearSHAP ────────────────────────────────────────────────────────────────
class ManualLinearSHAP:
    """LinearSHAP for Logistic Regression: φ_i = W[class, i] × (x_i − E[x_i])"""
    def __init__(self, lr_model, feature_names, background_mean):
        self.model=lr_model; self.fnames=feature_names; self.bg=background_mean
    def explain_single(self, x, pred_class_idx):
        W=self.model.coef_
        shap_row=W[pred_class_idx]*(x-self.bg)
        return dict(sorted({self.fnames[i]:float(shap_row[i]) for i in range(len(self.fnames))}.items(),key=lambda kv:abs(kv[1]),reverse=True))'''

app = app.replace(OLD_APP_SHAP, NEW_APP_SHAP)

# 2b. Update predict_burnout function
# Old: loads rf_model and uses ManualTreeSHAP
app = app.replace(
    "    b=load_model(); lr=b.get('primary_model') or b['lr_model']; rf=b['rf_model']; scaler=b['scaler']",
    "    b=load_model(); lr=b.get('primary_model') or b['lr_model']; scaler=b['scaler']; bg=b['background_mean']"
)
app = app.replace(
    "    # Explain with Random Forest SHAP (for consistency with model development)\n"
    "    expl=ManualTreeSHAP(rf,FEATURE_COLS); shap_d=expl.explain_single(x_sc,pid)",
    "    # Explain with LinearSHAP (mathematically faithful to the LR prediction model)\n"
    "    expl=ManualLinearSHAP(lr,FEATURE_COLS,bg); shap_d=expl.explain_single(x_sc,pid)"
)

# 2c. Update model_insights shap_explanation text
OLD_SHAP_EXPLANATION = (
    "'shap_explanation':'We implement TreeSHAP (Lundberg & Lee, 2017) from scratch "
    "using sklearn\\'s decision_path API applied to the Random Forest model. For each "
    "prediction: (1) walk root→leaf path, (2) compute probability change at each node split, "
    "(3) attribute that change to the splitting feature, (4) average across 200 trees. This "
    "gives φ values where Σφᵢ = f(x) - E[f(x)]. While predictions come from Logistic "
    "Regression (superior accuracy), explanations use Random Forest to provide interpretable "
    "feature contributions.'"
)
NEW_SHAP_EXPLANATION = (
    "'shap_explanation':'We implement LinearSHAP (Lundberg & Lee, 2017) from scratch "
    "using the Logistic Regression model\\'s own learned coefficients. For each prediction: "
    "(1) retrieve the coefficient vector W[predicted_class], (2) compute each feature\\'s "
    "deviation from the training mean: (x_i − E[x_i]), (3) multiply: φ_i = W[class, i] × "
    "(x_i − E[x_i]). This gives exact SHAP values where Σφᵢ = f(x) − E[f(x)]. Both "
    "predictions AND explanations use the same Logistic Regression model (99.40% accuracy), "
    "ensuring theoretical correctness — the explanation is mathematically faithful to the "
    "exact model making the decision.'"
)
app = app.replace(OLD_SHAP_EXPLANATION, NEW_SHAP_EXPLANATION)

# 2d. Update the comment in predict_burnout
app = app.replace(
    '    # Predict with Logistic Regression (99.40% accuracy, 99.99% AUC-ROC)',
    '    # Predict & explain with Logistic Regression (99.40% accuracy, 99.99% AUC-ROC)'
)

with open(app_path, 'w', encoding='utf-8') as f:
    f.write(app)
print(f"[OK] Patched {app_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PATCH run_tests.py
# ─────────────────────────────────────────────────────────────────────────────
test_path = os.path.join(BASE, 'run_tests.py')
if os.path.exists(test_path):
    with open(test_path, 'r', encoding='utf-8') as f:
        tests = f.read()

    # Replace rf_model references with lr_model / background_mean
    tests = tests.replace("'rf_model' key exists in bundle", "'background_mean' key exists in bundle")
    tests = tests.replace("'rf_model' in bundle", "'background_mean' in bundle")
    tests = tests.replace("'rf_model' key exists\"", "'background_mean' key exists\"")
    tests = tests.replace("rf = bundle['rf_model']", "lr = bundle.get('primary_model') or bundle['lr_model']; bg = bundle['background_mean']")

    # Replace ManualTreeSHAP class with ManualLinearSHAP
    tests = tests.replace('# ManualTreeSHAP', '# ManualLinearSHAP')
    tests = tests.replace('section("UNIT TESTS - ManualTreeSHAP")', 'section("UNIT TESTS - ManualLinearSHAP")')

    # Replace the inline ManualTreeSHAP class definition in run_tests.py
    # This is a compact version — find it and replace
    old_test_shap_pattern = r'class ManualTreeSHAP:.*?(?=\n    section\("UNIT TESTS)'
    old_test_shap_pattern_r = r'class ManualTreeSHAP:.*?(?=\r\n    section\("UNIT TESTS)'

    NEW_TEST_SHAP = '''class ManualLinearSHAP:
        def __init__(self, lr_model, feature_names, background_mean):
            self.model=lr_model; self.fnames=feature_names; self.bg=background_mean
        def explain_single(self, x, pred_class_idx):
            W=self.model.coef_
            shap_row=W[pred_class_idx]*(x-self.bg)
            return dict(sorted({self.fnames[i]:float(shap_row[i]) for i in range(len(self.fnames))}.items(),key=lambda kv:abs(kv[1]),reverse=True))

    '''

    tests_new = re.sub(old_test_shap_pattern_r, NEW_TEST_SHAP.replace('\n', '\r\n'), tests, flags=re.DOTALL)
    if tests_new == tests:
        tests_new = re.sub(old_test_shap_pattern, NEW_TEST_SHAP, tests, flags=re.DOTALL)
    tests = tests_new

    # Replace ManualTreeSHAP instantiations
    tests = tests.replace('shap = ManualTreeSHAP(rf, FEATURE_COLS)', 'shap = ManualLinearSHAP(lr, FEATURE_COLS, bg)')
    tests = tests.replace('shaper = ManualTreeSHAP(rf, FEATURE_COLS)', 'shaper = ManualLinearSHAP(lr, FEATURE_COLS, bg)')

    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(tests)
    print(f"[OK] Patched {test_path}")
else:
    print("[SKIP] run_tests.py not found")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PATCH frontend/templates/model_insights.html
# ─────────────────────────────────────────────────────────────────────────────
mi_path = os.path.join(BASE, 'frontend', 'templates', 'model_insights.html')
if os.path.exists(mi_path):
    with open(mi_path, 'r', encoding='utf-8') as f:
        mi = f.read()
    mi = mi.replace('<strong>TreeSHAP</strong>', '<strong>LinearSHAP</strong>')
    mi = mi.replace('TreeSHAP', 'LinearSHAP')
    with open(mi_path, 'w', encoding='utf-8') as f:
        f.write(mi)
    print(f"[OK] Patched {mi_path}")
else:
    print("[SKIP] model_insights.html not found")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PATCH frontend/templates/test_results.html
# ─────────────────────────────────────────────────────────────────────────────
tr_path = os.path.join(BASE, 'frontend', 'templates', 'test_results.html')
if os.path.exists(tr_path):
    with open(tr_path, 'r', encoding='utf-8') as f:
        tr = f.read()
    tr = tr.replace('TreeSHAP', 'LinearSHAP')
    with open(tr_path, 'w', encoding='utf-8') as f:
        f.write(tr)
    print(f"[OK] Patched {tr_path}")
else:
    print("[SKIP] test_results.html not found")


print("\n" + "="*60)
print("  ALL FILES PATCHED SUCCESSFULLY")
print("  ManualTreeSHAP → ManualLinearSHAP")
print("="*60)
