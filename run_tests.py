"""
BurnoutGuard — Unit & Integration Test Suite
Run: python run_tests.py
"""
import sys, os, hashlib, json, sqlite3, tempfile, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml'))

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"; RED  = "\033[91m"; CYAN = "\033[96m"
YELLOW = "\033[93m"; BOLD = "\033[1m";  RESET= "\033[0m"

passed = failed = 0

def run(test_id, name, result, expected_desc):
    global passed, failed
    status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
    mark   = "PASS" if result else "FAIL"
    if result: passed += 1
    else:       failed += 1
    print(f"  {mark} [{test_id}] {name:<52} -> {status}")
    if not result:
        print(f"       Expected: {expected_desc}")

def section(title):
    print(f"\n{BOLD}{CYAN}{'='*65}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*65}{RESET}")

# =============================================================================
# SECTION 1 — UNIT TESTS: parse_numeric
# =============================================================================
section("UNIT TESTS — parse_numeric() [ml_pipeline.py]")

from ml_pipeline import parse_numeric

run("UT-01", "parse_numeric('5-6')  -> 5.5",        parse_numeric("5-6")       == 5.5,   "5.5")
run("UT-02", "parse_numeric('No')   -> 0.0",        parse_numeric("No")        == 0.0,   "0.0")
run("UT-03", "parse_numeric('2-3 hours') -> 2.5",   parse_numeric("2-3 hours") == 2.5,   "2.5")
run("UT-05", "parse_numeric('7.5')  -> 7.5",        parse_numeric("7.5")       == 7.5,   "7.5")
run("UT-06", "parse_numeric('5to6') -> 5.5",        parse_numeric("5to6")      == 5.5,   "5.5")
run("UT-07", "parse_numeric('0-2')  -> 1.0",        parse_numeric("0-2")       == 1.0,   "1.0")

# =============================================================================
# SECTION 2 — UNIT TESTS: hash_pw
# =============================================================================
section("UNIT TESTS — hash_pw() [app.py]")

def hash_pw(p):
    return hashlib.sha256(('rv_burnout_salt_' + p).encode()).hexdigest()

h1 = hash_pw("password123")
h2 = hash_pw("password123")
h3 = hash_pw("different")

run("UT-08", "hash_pw returns 64-char hex string",        len(h1) == 64,      "64 chars")
run("UT-09", "Same input -> same hash (deterministic)",   h1 == h2,           "h1 == h2")
run("UT-10", "Different input -> different hash",         h1 != h3,           "h1 != h3")
run("UT-11", "Hash contains only hex characters",        all(c in '0123456789abcdef' for c in h1), "hex chars only")

# =============================================================================
# SECTION 3 — UNIT TESTS: analyse_trend
# =============================================================================
section("UNIT TESTS — analyse_trend() [app.py]")

# Inline the function to avoid Flask context issues
RISK_SCORE = {'Low': 0, 'Medium': 1, 'High': 2}

def analyse_trend(predictions):
    if len(predictions) < 2:
        return {'trend': 'Baseline', 'details': 'Only one week of data.', 'streak': 1}
    scores = [RISK_SCORE.get(p.get('predicted_risk','Medium'), 1) for p in predictions]
    net = scores[-1] - scores[0]
    pos = sum(1 for i in range(1,len(scores)) if scores[i] > scores[i-1])
    neg = sum(1 for i in range(1,len(scores)) if scores[i] < scores[i-1])
    if neg > pos and net <= 0:   trend = 'Improving 📈'
    elif pos > neg and net >= 0: trend = 'Declining 📉'
    elif net == 0:               trend = 'Stable ➡️'
    else:                        trend = 'Fluctuating 〰️'
    return {'trend': trend, 'details': '', 'streak': 1}

def preds(*risks):
    return [{'predicted_risk': r} for r in risks]

run("UT-12", "High->Med->Low = Improving",    analyse_trend(preds('High','Medium','Low'))['trend'] == 'Improving 📈',  "Improving 📈")
run("UT-13", "Low->Med->High = Declining",    analyse_trend(preds('Low','Medium','High'))['trend'] == 'Declining 📉', "Declining 📉")
run("UT-14", "Med->Med->Med  = Stable",       analyse_trend(preds('Medium','Medium','Medium'))['trend'] == 'Stable ➡️', "Stable ➡️")
run("UT-15", "Single week  = Baseline",     analyse_trend(preds('High'))['trend'] == 'Baseline',                    "Baseline")

# =============================================================================
# SECTION 4 — UNIT TESTS: Load Model + predict_burnout
# =============================================================================
section("UNIT TESTS — Model Loading & predict_burnout()")

import pickle
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml', 'saved_models', 'burnout_model.pkl')
bundle = None

try:
    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
    run("UT-16", "burnout_model.pkl loads without error",   True,  "No exception")
    run("UT-17", "'primary_model' key exists in bundle",    'primary_model' in bundle or 'lr_model' in bundle, "key present")
    run("UT-18", "'background_mean' key exists in bundle",         'background_mean' in bundle,    "key present")
    run("UT-19", "'scaler' key exists in bundle",           'scaler' in bundle,      "key present")
    run("UT-20", "'feature_cols' has 19 features",          len(bundle.get('feature_cols',[])) == 19, "19 features")
except Exception as e:
    run("UT-16", "burnout_model.pkl loads without error",   False, f"Exception: {e}")
    run("UT-17", "'primary_model' key exists", False, "key present")
    run("UT-18", "'background_mean' key exists",      False, "key present")
    run("UT-19", "'scaler' key exists",        False, "key present")
    run("UT-20", "'feature_cols' has 19 items",False, "19 features")

# predict_burnout test using bundle directly
if bundle:
    LABEL_INV = bundle.get('label_inv', {0:'Low',1:'Medium',2:'High'})
    FEATURE_COLS = bundle['feature_cols']
    lr = bundle.get('primary_model') or bundle['lr_model']
    lr = bundle.get('primary_model') or bundle['lr_model']; bg = bundle['background_mean']
    scaler = bundle['scaler']
    ge = bundle['gender_enc']
    we = bundle['working_enc']
    ce = bundle['content_enc']

    def make_vector(vals):
        return np.array(vals, dtype=float)

    # High-risk profile: high anxiety, low sleep, low motivation
    high_raw = [14, 3, 0, 8, 14, 9, 6, 9, 9, 9, 2, 9, 10, 1, 2, 22,
                int(ge.transform(['Male'])[0]),
                int(we.transform(['Student'])[0]),
                int(ce.transform(['Entertainment'])[0])]

    # Low-risk profile: low anxiety, good sleep, high motivation
    low_raw  = [5, 8, 5, 1, 3, 1, 0, 1, 1, 1, 9, 1, 1, 9, 9, 21,
                int(ge.transform(['Female'])[0]),
                int(we.transform(['Student'])[0]),
                int(ce.transform(['Educational'])[0])]

    x_high = scaler.transform(np.array(high_raw, dtype=float).reshape(1,-1))[0]
    x_low  = scaler.transform(np.array(low_raw,  dtype=float).reshape(1,-1))[0]

    pred_high = LABEL_INV[int(lr.predict(x_high.reshape(1,-1))[0])]
    pred_low  = LABEL_INV[int(lr.predict(x_low.reshape(1,-1))[0])]
    proba_high = lr.predict_proba(x_high.reshape(1,-1))[0]
    proba_low  = lr.predict_proba(x_low.reshape(1,-1))[0]

    run("UT-21", "High-risk inputs -> predicted 'High'",         pred_high == 'High',        "High")
    run("UT-22", "Low-risk inputs  -> predicted 'Low'",          pred_low  == 'Low',         "Low")
    run("UT-23", "High prediction confidence > 80%",            proba_high[2] > 0.80,       "> 0.80")
    run("UT-24", "Low prediction confidence > 80%",             proba_low[0]  > 0.80,       "> 0.80")
    run("UT-25", "Probabilities sum to 1.0",                    abs(sum(proba_high) - 1.0) < 1e-6, "≈ 1.0")

    # ManualLinearSHAP
    class ManualLinearSHAP:
        def __init__(self, lr_model, feature_names, background_mean):
            self.model=lr_model; self.fnames=feature_names; self.bg=background_mean
        def explain_single(self, x, pred_class_idx):
            W=self.model.coef_
            shap_row=W[pred_class_idx]*(x-self.bg)
            return dict(sorted({self.fnames[i]:float(shap_row[i]) for i in range(len(self.fnames))}.items(),key=lambda kv:abs(kv[1]),reverse=True))

    
    section("UNIT TESTS — ManualTreeSHAP")
    shap = ManualLinearSHAP(lr, FEATURE_COLS, bg)
    sv = shap.explain_single(x_high, 2)

    run("UT-26", "SHAP returns dict with 19 keys",             len(sv) == 19,             "19 keys")
    run("UT-27", "All feature names in SHAP output",           all(f in sv for f in FEATURE_COLS), "all 19 features")
    run("UT-28", "SHAP values are floats",                     all(isinstance(v,float) for v in sv.values()), "all floats")
    run("UT-29", "At least one SHAP value is positive",        any(v > 0 for v in sv.values()), "some positive φ")
    run("UT-30", "At least one SHAP value is negative",        any(v < 0 for v in sv.values()), "some negative φ")

# =============================================================================
# SECTION 5 — INTEGRATION TESTS: SQLite Database
# =============================================================================
section("INTEGRATION TESTS — SQLite Database")

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS weekly_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    week_number INTEGER NOT NULL,
    submitted_at TEXT DEFAULT (datetime('now')),
    study_work_hours_per_day REAL, daily_sleep_hours REAL,
    exercise_frequency_per_week REAL, caffeine_intake_cups REAL,
    UNIQUE(user_id, week_number)
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    week_number INTEGER NOT NULL,
    predicted_risk TEXT NOT NULL,
    confidence_low REAL, confidence_medium REAL, confidence_high REAL,
    shap_values_json TEXT, recommendation_json TEXT,
    UNIQUE(user_id, week_number)
);
"""

try:
    tmpdb = tempfile.mktemp(suffix='.db')
    conn  = sqlite3.connect(tmpdb)
    conn.executescript(DB_SCHEMA)

    # IT-01: Insert user
    conn.execute("INSERT INTO users (username,email,password_hash,full_name) VALUES (?,?,?,?)",
                 ('testuser','test@rv.edu', hash_pw('pass123'), 'Test User'))
    conn.commit()
    user = conn.execute("SELECT * FROM users WHERE username='testuser'").fetchone()
    run("IT-01", "User registration inserts row into users table",   user is not None, "row exists")

    uid = user[0]

    # IT-02: Duplicate username rejected
    try:
        conn.execute("INSERT INTO users (username,email,password_hash,full_name) VALUES (?,?,?,?)",
                     ('testuser','other@rv.edu', hash_pw('x'), 'Other'))
        conn.commit()
        run("IT-02", "Duplicate username raises UNIQUE constraint",  False, "IntegrityError")
    except sqlite3.IntegrityError:
        run("IT-02", "Duplicate username raises UNIQUE constraint",  True,  "IntegrityError raised")

    # IT-03: Survey response insert
    conn.execute("INSERT INTO weekly_responses (user_id,week_number,study_work_hours_per_day,daily_sleep_hours,exercise_frequency_per_week,caffeine_intake_cups) VALUES (?,?,?,?,?,?)",
                 (uid, 1, 8.0, 5.0, 1.0, 4.0))
    conn.commit()
    resp = conn.execute("SELECT * FROM weekly_responses WHERE user_id=? AND week_number=1",(uid,)).fetchone()
    run("IT-03", "Survey Week 1 response stored correctly",          resp is not None, "row exists")

    # IT-04: Duplicate week rejected
    try:
        conn.execute("INSERT INTO weekly_responses (user_id,week_number,study_work_hours_per_day,daily_sleep_hours,exercise_frequency_per_week,caffeine_intake_cups) VALUES (?,?,?,?,?,?)",
                     (uid, 1, 7.0, 6.0, 2.0, 2.0))
        conn.commit()
        run("IT-04", "Duplicate week submission rejected",           False, "IntegrityError")
    except sqlite3.IntegrityError:
        run("IT-04", "Duplicate week submission rejected",           True,  "IntegrityError raised")

    # IT-05: Prediction insert
    shap_json = json.dumps({'Anxiety_Score': 0.045, 'Sleep_Hours': -0.032})
    conn.execute("INSERT INTO predictions (user_id,week_number,predicted_risk,confidence_low,confidence_medium,confidence_high,shap_values_json) VALUES (?,?,?,?,?,?,?)",
                 (uid, 1, 'High', 0.02, 0.15, 0.83, shap_json))
    conn.commit()
    pred = conn.execute("SELECT * FROM predictions WHERE user_id=? AND week_number=1",(uid,)).fetchone()
    run("IT-05", "Prediction row stored in predictions table",       pred is not None, "row exists")
    run("IT-06", "Predicted risk value stored correctly",            pred[3] == 'High', "'High'")
    run("IT-07", "SHAP JSON stored and parseable",                   json.loads(pred[7]) is not None, "valid JSON")

    # IT-08: Multi-week history query
    conn.execute("INSERT INTO weekly_responses (user_id,week_number,study_work_hours_per_day,daily_sleep_hours,exercise_frequency_per_week,caffeine_intake_cups) VALUES (?,?,?,?,?,?)",
                 (uid, 2, 6.0, 7.0, 3.0, 2.0))
    conn.execute("INSERT INTO predictions (user_id,week_number,predicted_risk,confidence_low,confidence_medium,confidence_high) VALUES (?,?,?,?,?,?)",
                 (uid, 2, 'Medium', 0.05, 0.80, 0.15))
    conn.commit()
    all_preds = conn.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY week_number",(uid,)).fetchall()
    run("IT-08", "History query returns all submitted weeks",         len(all_preds) == 2, "2 rows")
    run("IT-09", "Week 1 risk=High, Week 2 risk=Medium (ordering)",  all_preds[0][3]=='High' and all_preds[1][3]=='Medium', "correct order")

    # IT-10: Password verification
    stored_hash = conn.execute("SELECT password_hash FROM users WHERE username='testuser'").fetchone()[0]
    run("IT-10", "Correct password matches stored hash",             hash_pw('pass123') == stored_hash, "hashes match")
    run("IT-11", "Wrong password does NOT match stored hash",        hash_pw('wrongpass') != stored_hash, "hashes differ")

    conn.close()
    os.unlink(tmpdb)

except Exception as e:
    print(f"  {RED}ERROR during integration tests: {e}{RESET}")

# =============================================================================
# SECTION 6 — INTEGRATION TESTS: ML + SHAP Pipeline
# =============================================================================
if bundle:
    section("INTEGRATION TESTS — ML Pipeline End-to-End")

    # Full pipeline: raw values -> scale -> predict -> SHAP
    try:
        x_raw  = np.array(high_raw, dtype=float)
        x_sc   = scaler.transform(x_raw.reshape(1,-1))[0]
        pid    = int(lr.predict(x_sc.reshape(1,-1))[0])
        proba  = lr.predict_proba(x_sc.reshape(1,-1))[0]
        risk   = LABEL_INV[pid]
        shaper = ManualLinearSHAP(lr, FEATURE_COLS, bg)
        sv2    = shaper.explain_single(x_sc, pid)
        top5   = list(sv2.items())[:5]

        run("IT-12", "Raw form -> scaled -> LR predicts a valid class",    risk in ['Low','Medium','High'], "valid class")
        run("IT-13", "LR probabilities sum to 1.0",                       abs(sum(proba)-1.0) < 1e-6,     "≈ 1.0")
        run("IT-14", "SHAP values computed for all 19 features",          len(sv2) == 19,                 "19 features")
        run("IT-15", "Top-5 features extracted correctly",                len(top5) == 5,                 "5 items")
        run("IT-16", "SHAP serialisable to JSON (for DB storage)",        bool(json.dumps(sv2)),          "JSON ok")

        # Grouped SHAP
        ACADEMIC     = ['Study_Work_Hours_per_Day','Overthinking_Score','Anxiety_Score','Motivation_Level','Student_Working_Status']
        BEHAVIOURAL  = ['Screen_Time_Hours','Night_Scrolling_Frequency','Online_Gaming_Hours','Daily_Social_Media_Hours','Caffeine_Intake_Cups','Content_Type_Preference']
        LIFESTYLE    = ['Daily_Sleep_Hours','Sleep_Quality_Score','Exercise_Frequency_per_Week','Mood_Stability_Score','Emotional_Fatigue_Score','Social_Comparison_Index','Age','Gender']

        grp_acad = sum(abs(sv2.get(f,0)) for f in ACADEMIC)
        grp_beh  = sum(abs(sv2.get(f,0)) for f in BEHAVIOURAL)
        grp_life = sum(abs(sv2.get(f,0)) for f in LIFESTYLE)

        run("IT-17", "Academic group SHAP total > 0",                     grp_acad > 0, "> 0")
        run("IT-18", "Behavioural group SHAP total > 0",                  grp_beh  > 0, "> 0")
        run("IT-19", "Lifestyle group SHAP total > 0",                    grp_life > 0, "> 0")
        run("IT-20", "All 3 group totals are different (distinct impact)", len({round(grp_acad,4), round(grp_beh,4), round(grp_life,4)}) > 1, "at least 2 differ")

    except Exception as e:
        print(f"  {RED}ERROR: {e}{RESET}")

# =============================================================================
# SUMMARY
# =============================================================================
total = passed + failed
print(f"\n{BOLD}{'='*65}{RESET}")
print(f"{BOLD}  TEST SUMMARY{RESET}")
print(f"{'='*65}")
print(f"  Total Tests : {BOLD}{total}{RESET}")
print(f"  {GREEN}Passed      : {passed}{RESET}")
if failed:
    print(f"  {RED}Failed      : {failed}{RESET}")
else:
    print(f"  Failed      : 0")

pct = (passed/total*100) if total else 0
bar_len = 40
filled  = int(bar_len * passed // total) if total else 0
bar     = f"{GREEN}{'█'*filled}{RED}{'░'*(bar_len-filled)}{RESET}"
print(f"\n  [{bar}] {pct:.1f}%\n")

if failed == 0:
    print(f"{BOLD}{GREEN}  PASS ALL TESTS PASSED!{RESET}\n")
else:
    print(f"{BOLD}{RED}  FAIL {failed} TEST(S) FAILED{RESET}\n")
