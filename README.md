# 🔥 BurnoutGuard: Student Burnout Risk Prediction System

**An AI-Driven Mental Health Assessment Tool with Explainable Predictions**

> Student burnout is a growing mental health crisis in Indian universities. BurnoutGuard provides early detection through machine learning, explainability through SHAP, and actionable recovery plans.

**Project:** RV University — BTech CSE Final Year (4 Credits)  
**Student:** Nandini | **Year:** 2024–25  
**Status:** ✅ Complete & Production-Ready

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Database Schema](#database-schema)
- [ML Pipeline](#ml-pipeline)
- [SHAP Explainability](#shap-explainability)
- [Results & Metrics](#results--metrics)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation (3 Steps)
```bash
# 1. Clone repository
git clone <repo-url>
cd BurnoutGuard_MiniProject

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the application
python app.py
```

**Access the app:** http://localhost:5000

---

## 🎯 Project Overview

### Problem Statement
Student burnout affects mental health, academic performance, and dropout rates. Current solutions lack:
- **Early detection mechanisms** — no objective assessment tools
- **Explainability** — ML models act as "black boxes"
- **Temporal tracking** — no trend detection across time
- **Personalization** — generic advice instead of data-driven interventions

### Our Solution
**BurnoutGuard** combines three core innovations:
1. **Predictive ML Model** — Logistic Regression for production prediction (probabilities) + Random Forest retained for explanations
2. **Explainable AI** — Custom TreeSHAP-style implementation (no external `shap` dependency)
3. **Temporal Tracking** — 4-week progression monitoring with trend alerts
4. **Personalized Plans** — Recovery recommendations targeted at identified risk drivers

### Target Audience
- Undergraduate & postgraduate students (age 18-25)
- University counseling services
- Mental health institutions

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🔐 Secure Authentication** | SHA-256 password hashing, session management |
| **📋 Weekly Wellness Survey** | Weekly survey submission (Week 1–4) with sequential tracking |
| **🤖 ML Prediction** | Logistic Regression prediction: Low/Medium/High with confidence probabilities |
| **💡 SHAP Explanations** | Feature-level breakdown grouped by life domain (Academic/Behavioral/Lifestyle) |
| **📊 4-Week Tracking** | Temporal trend visualization with risk progression |
| **🎯 Personalized Plans** | Recovery recommendations based on risk drivers |
| **📧 Email Alerts** | Automated notifications (welcome, results, trend warnings) |
| **📈 Model Insights** | Confusion matrices, AUC-ROC, feature importance, model comparison |

---

## 🛠️ Technology Stack

### Backend & ML
- **Language:** Python 3.10
- **Web Framework:** Flask 2.3+
- **ML Libraries:** scikit-learn 1.3+, pandas 2.0+, numpy 1.24+
- **Visualization:** matplotlib 3.7+, seaborn 0.12+

### Frontend
- **Markup:** HTML5
- **Styling:** CSS3
- **Interactivity:** JavaScript (vanilla)
- **Responsive Design:** Mobile-first CSS

### Database
- **Engine:** SQLite 3
- **Schema:** 3 main tables (users, weekly_responses, predictions)

### Deployment
- **Local:** Flask development server
- **Production-Ready:** Gunicorn, Docker, Cloud (AWS/Heroku)

---

## 🏗️ System Architecture



### 5-Layer Architecture

```
┌──────────────────────────────────────────────────────┐
│         FRONTEND LAYER (User Interface)               │
│  HTML5 + CSS3 + JavaScript (Responsive Design)        │
│  Pages: Landing, Login, Survey, Results, Dashboard    │
└─────────────────────────┬──────────────────────────────┘
                          │ HTTP/REST API
┌─────────────────────────▼──────────────────────────────┐
│      APPLICATION LAYER (Flask Backend - app.py)        │
│  • Authentication Service (SHA-256 hashing)            │
│  • Survey Handler (validation, persistence)            │
│  • Prediction Engine (ML inference)                    │
│  • SHAP Explainer (feature contributions)              │
│  • Visualization Engine (matplotlib charts)            │
│  • Email Service (SMTP notifications)                  │
│  • Trend Analyzer (4-week progression)                │
└─────────────────────────┬──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│      MACHINE LEARNING LAYER (ml/ml_pipeline.py)        │
│  • Logistic Regression (production prediction model)   │
│  • Random Forest (SHAP-style explainability model)     │
│  • Model comparisons: DT, KNN, SVM, Naive Bayes        │
│  • StandardScaler (feature normalization)              │
│  • Label Encoders (categorical variables)              │
│  • TreeSHAP Calculator (custom implementation)         │
│  • Visualization Generator                             │
└─────────────────────────┬──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│        DATA LAYER (Database & Model Artifacts)         │
│  • SQLite Database (burnout.db)                        │
│    ├── users table                                     │
│    ├── weekly_responses table                          │
│    └── predictions table                               │
│  • Model Files (saved_models/)                         │
│    ├── burnout_model.pkl (LR + RF + scaler + encoders) │
│    ├── metrics.json (evaluation results)               │
│    └── model_comparison_results.json                   │
└──────────────────────────────────────────────────────────┘
```

---

## 📦 Installation & Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/your-repo/burnoutguard.git
cd BurnoutGuard_MiniProject
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using Python venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- flask>=2.3.0
- scikit-learn>=1.3.0
- pandas>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- openpyxl>=3.1.0 (for Excel file support)

### Step 4: Set Up Environment Variables (Optional)
Create a `.env` file in the project root:
```
FLASK_ENV=development
BURNOUT_MAIL_SENDER=your_email@gmail.com
BURNOUT_MAIL_PASSWORD=your_16_char_app_password
```

### Step 5: Train ML Model (First Time)
```bash
python ml/ml_pipeline.py
```
This generates:
- `ml/saved_models/burnout_model.pkl` — trained models
- `ml/saved_models/metrics.json` — evaluation metrics
- `frontend/static/img/*.png` — visualization charts

### Step 6: Initialize Database
The SQLite database is auto-created on first run of `app.py`.

### Step 7: Start Flask Server
```bash
python app.py
```

**Output:**
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### Step 8: Access Web Application
Open browser → http://localhost:5000

---

## 💻 Usage Guide

### User Journey

#### 1. Landing Page (`http://localhost:5000`)
- Overview of BurnoutGuard
- Features highlight
- Call-to-action buttons (Login/Register)

#### 2. Registration (`/register`)
- Fill form: username, email, password, gender, working status
- Password hashed with SHA-256
- Account created in database
- Welcome email sent

#### 3. Login (`/login`)
- Enter username and password
- Session created with 24-hour timeout
- Redirect to survey page

#### 4. Weekly Survey (`/survey`)
**20 Questions across 4 weeks:**
- Academic factors (5 Qs): study hours, sleep, caffeine
- Behavioral factors (7 Qs): anxiety, overthinking, mood, fatigue
- Lifestyle factors (10 Qs): screen time, exercise, social media, age, gender

**Format:** Numeric sliders (0-10 scale)  
**Time to complete:** ~5 minutes

#### 5. Prediction Results (`/result`)
- **Prediction badge:** Low / Medium / High risk
- **Confidence scores:** % for each category
- **SHAP explanation:** Top 5 features grouped by domain
- **Waterfall plot:** Visual feature contributions
- **Recovery plan:** Personalized recommendations

#### 6. Dashboard (`/dashboard`)
- View all 4 weeks of predictions
- Trend line chart (risk evolution)
- Trend interpretation (Stable/Improving/Declining)
- Risk progression alerts

#### 7. History (`/history`)
- Detailed prediction breakdown per week
- SHAP values per submission
- Recommendation history
- Download results (optional)

#### 8. Model Insights (`/model_insights`)
- Confusion matrices (RF vs LR)
- ROC-AUC curves
- Feature importance ranking
- Model comparison metrics

---

## 📁 Project Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                              │
│   Landing → Register → Login → Survey → Result → Dashboard      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP (Flask routes)
┌───────────────────────────▼─────────────────────────────────────┐
│                     FLASK BACKEND (app.py)                       │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Auth System │  │ Survey Routes│  │  Prediction Engine   │  │
│  │  (SHA-256)   │  │  (4 weeks)   │  │  (LR + SHAP + Rec)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Trend Engine │  │  Viz Engine  │  │   Recommendation     │  │
│  │ (Pure Logic) │  │ (matplotlib) │  │   Engine (Rules)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
              ┌─────────────┴──────────────┐
┌─────────────▼──────┐          ┌──────────▼────────────┐
│  SQLite Database   │          │    ML Model Bundle    │
│  ─────────────     │          │  ──────────────────   │
│  users             │          │  Logistic Regression  │
│  weekly_responses  │          │  Random Forest (RF)   │
│  predictions       │          │  StandardScaler       │
└────────────────────┘          │  LabelEncoders        │
                                │  ManualTreeSHAP       │
                                └───────────────────────┘
```

---



```
BurnoutGuard_MiniProject/
│
├── 📄 README.md                          ← This file
├── 📄 requirements.txt                   ← Python dependencies
├── 🔐 .env                              ← Environment variables (optional)
│
├── 🐍 app.py                            ← Flask backend (1500+ lines)
│   ├── Authentication routes (/register, /login, /logout)
│   ├── Survey routes (/survey, /submit_survey)
│   ├── Prediction & SHAP routes (/result, /dashboard, /history)
│   ├── Model insights routes (/model_insights)
│   ├── Error handling & logging
│   └── Email service integration
│
├── 📧 email_service.py                  ← Email notification system
│   ├── Gmail SMTP configuration
│   ├── Email templates (welcome, results, alerts)
│   ├── Scheduler for reminder emails
│   └── Event-driven notification triggers
│
├── 📂 ml/                               ← Machine Learning Pipeline
│   ├── 🐍 ml_pipeline.py               ← Model training & evaluation
│   │   ├── Data loading (synthetic + real)
│   │   ├── Feature engineering
│   │   ├── Model training (LR, RF, DT, KNN, SVM, NB)
│   │   ├── Cross-validation
│   │   ├── SHAP computation
│   │   ├── Visualization generation
│   │   └── Metrics calculation
│   │
│   ├── 📊 genz_mental_wellness_synthetic_dataset.csv  ← Training data (10K rows)
│   │
│   ├── 📂 saved_models/                 ← Model artifacts
│   │   ├── burnout_model.pkl            ← Model bundle (LR prediction + RF explainability + scaler + encoders)
│   │   ├── metrics.json                 ← Evaluation metrics
│   │   └── model_comparison_results.json ← RF vs LR comparison
│   │
│   └── 📝 Student Wellness Survey (Responses).xlsx  ← Real test data (optional; path set via TEST_XLSX in .env)
│
├── 📂 database/                         ← Data persistence
│   └── 🗄️ burnout.db                  ← SQLite database (auto-created)
│       ├── users table
│       ├── weekly_responses table
│       └── predictions table
│
├── 📂 frontend/                         ← Web UI (HTML/CSS/JavaScript)
│   │
│   ├── 📂 templates/                   ← HTML pages
│   │   ├── base.html                   ← Base layout (navbar, footer, styles)
│   │   ├── index.html                  ← Landing page
│   │   ├── login.html                  ← Login form
│   │   ├── register.html               ← Registration form
│   │   ├── survey.html                 ← Weekly survey form (Week 1–4)
│   │   ├── result.html                 ← Prediction result + SHAP explanation
│   │   ├── dashboard.html              ← Main dashboard (4-week tracking)
│   │   ├── history.html                ← Detailed week-by-week breakdown
│   │   ├── model_insights.html         ← ML evaluation & metrics
│   │   └── test_results.html           ← Test results on 52 real submissions
│   │
│   └── 📂 static/                      ← Static assets
│       ├── 🖼️ img/                     ← Charts & visualizations (pipeline + runtime)
│       ├── 📦 manifest/                ← PWA manifest
│       │   └── manifest.json
│       └── sw.js                       ← Service Worker
```

---

## 💾 Database Schema

### users
| Column        | Type    | Description              |
|---------------|---------|--------------------------|
| id            | INTEGER | Primary key              |
| username      | TEXT    | Unique login name        |
| email         | TEXT    | Unique email             |
| password_hash | TEXT    | SHA-256 hashed password  |
| full_name     | TEXT    | Display name             |
| created_at    | TEXT    | Timestamp                |

### weekly_responses
| Column                      | Type    | Description                  |
|-----------------------------|---------|------------------------------|
| id                          | INTEGER | Primary key                  |
| user_id                     | INTEGER | FK → users.id                |
| week_number                 | INTEGER | 1–4                          |
| submitted_at                | TEXT    | Submission timestamp         |
| study_work_hours_per_day    | REAL    | Survey feature               |
| daily_sleep_hours           | REAL    | Survey feature               |
| … (17 more feature columns) |         |                              |

### predictions
| Column              | Type    | Description                          |
|---------------------|---------|--------------------------------------|
| id                  | INTEGER | Primary key                          |
| user_id             | INTEGER | FK → users.id                        |
| week_number         | INTEGER | 1–4                                  |
| response_id         | INTEGER | FK → weekly_responses.id             |
| predicted_risk      | TEXT    | Low / Medium / High                  |
| confidence_low      | REAL    | P(Low) from production model         |
| confidence_medium   | REAL    | P(Medium) from production model      |
| confidence_high     | REAL    | P(High) from production model        |
| shap_values_json    | TEXT    | JSON: {feature: shap_value, ...}     |
| top_features_json   | TEXT    | JSON: top 5 feature contributions    |
| grouped_shap_json   | TEXT    | JSON: grouped SHAP by domain         |
| recommendation_json | TEXT    | JSON: full personalised plan         |
| predicted_at        | TEXT    | Timestamp                            |

---



### Tables Overview

#### users
Stores user account information.
```sql
CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,           -- SHA-256 hashed (+ salt in app logic)
  full_name TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### weekly_responses
Stores student survey responses (weekly, Week 1–4).
```sql
CREATE TABLE weekly_responses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  week_number INTEGER NOT NULL,          -- 1, 2, 3, or 4
  
  -- Survey Features Stored (as implemented in app.py)
  study_work_hours_per_day REAL,
  daily_sleep_hours REAL,
  exercise_frequency_per_week REAL,
  caffeine_intake_cups REAL,
  screen_time_hours REAL,
  night_scrolling_frequency REAL,
  online_gaming_hours REAL,
  overthinking_score REAL,
  social_comparison_index REAL,
  anxiety_score REAL,
  mood_stability_score REAL,
  emotional_fatigue_score REAL,
  daily_social_media_hours REAL,
  motivation_level REAL,
  sleep_quality_score REAL,
  age INTEGER,
  gender TEXT,
  working_status TEXT,
  content_type TEXT,
  
  submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
  UNIQUE(user_id, week_number)
);
```

#### predictions
Stores ML predictions, confidence scores, and SHAP explanations.
```sql
CREATE TABLE predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  week_number INTEGER NOT NULL,
  response_id INTEGER NOT NULL,
  
  predicted_risk TEXT NOT NULL,          -- 'Low', 'Medium', 'High'
  confidence_low REAL,                   -- P(Low)
  confidence_medium REAL,                -- P(Medium)
  confidence_high REAL,                  -- P(High)
  
  shap_values_json TEXT,                 -- JSON: {feature: shap_value}
  top_features_json TEXT,                -- JSON: top features list
  grouped_shap_json TEXT,                -- JSON: grouped by domain
  recommendation_json TEXT,              -- JSON: personalised plan
  predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY(response_id) REFERENCES weekly_responses(id) ON DELETE CASCADE,
  UNIQUE(user_id, week_number)
);
```

---

## 🧠 ML Pipeline

### Training Data
- **Source:** GenZ Mental Wellness Synthetic Dataset
- **Size:** 10,000 rows × 22 features
- **Target:** Burnout Risk (Low/Medium/High)
- **Class Distribution:**
  - Low: 64 samples
  - Medium: 4,548 samples
  - High: 5,388 samples

### Test Data (Real Submissions)
- **Source:** Google Form responses from actual students
- **Size:** Dynamic (latest count saved in `ml/saved_models/metrics.json`)
- **Data Quality:** Cleaned, outliers handled, categoricals encoded

### Features Used in the Web App (19 Total)
The production system uses the following 19 input features:
- Study/Work Hours per Day
- Daily Sleep Hours
- Exercise Frequency per Week
- Caffeine Intake (cups/day)
- Total Screen Time (hours/day)
- Night Scrolling Frequency
- Online Gaming Hours
- Overthinking Score
- Social Comparison Index
- Anxiety Score
- Mood Stability Score
- Emotional Fatigue Score
- Daily Social Media Hours
- Motivation Level
- Sleep Quality Score
- Age
- Gender (encoded)
- Working Status (encoded)
- Content Type Preference (encoded)

### Models Trained (ML Pipeline)
The pipeline trains and evaluates:
**Logistic Regression, Random Forest, Decision Tree, KNN, SVM, Naive Bayes**.

### Production Model Choice (as implemented in `app.py`)
- **Logistic Regression** is used for prediction and confidence probabilities.
- **Random Forest** is used for SHAP-style explainability (tree-path contributions).

### Latest Validation Metrics (from `ml/saved_models/metrics.json`)
| Model               | Accuracy | F1-Score | AUC-ROC |
|---------------------|----------|----------|---------|
| Logistic Regression | 99.40%   | 99.42%   | 99.99%  |
| Random Forest       | 96.05%   | 96.03%   | 99.57%  |

### Real Data Test Results
Real-response prediction distribution and per-student confidence values are stored in
`ml/saved_models/metrics.json` and displayed in the `/test_results` page.

---

## 💡 SHAP Explainability

### What is SHAP?
**SHapley Additive exPlanations** — a game-theory approach to explaining ML predictions at the feature level.

### Implementation
Custom TreeSHAP using sklearn's `decision_path` API (no external SHAP dependency).

**How it works:**
1. For each tree in Random Forest, trace decision path from root → leaf
2. At each split node, calculate feature contribution
3. Average contributions across all 200 trees
4. Result: φ_i = contribution of feature i to final prediction

**Interpretation:**
- φ_i > 0 → feature pushed prediction toward **higher burnout risk**
- φ_i < 0 → feature pushed prediction toward **lower burnout risk**
- |φ_i| larger → stronger influence

### User-Facing Output
1. **Waterfall Plot** — Top 5 features with +/- impacts
2. **Domain Grouping** — Features grouped by:
   - Academic factors
   - Behavioral factors
   - Lifestyle factors
3. **Text Explanation** — "Your anxiety (+0.28) and low sleep (-0.15) increased your risk by 0.40"

---

## 📊 Results & Metrics

### Model Performance Summary
Validation metrics are exported by the pipeline into:
`ml/saved_models/metrics.json`

**Latest validation results (from `metrics.json`):**
- **Logistic Regression (Production prediction model)**:
  - Accuracy: **99.40%**
  - F1-Score (weighted): **99.42%**
  - AUC-ROC (weighted OvR): **99.99%**
- **Random Forest (Explainability model for SHAP-style contributions)**:
  - Accuracy: **96.05%**
  - F1-Score (weighted): **96.03%**
  - AUC-ROC (weighted OvR): **99.57%**

### Explainability Outputs (SHAP)
SHAP-style explanations are computed **per student per week**, so the top drivers vary by individual.
The UI shows:
1. Top contributing factors (direction: increases / decreases risk)
2. Domain-grouped breakdown (Academic / Behavioural / Lifestyle)
3. Grouped SHAP chart image per result page

### Confusion Matrices
Generated and saved as PNG visualizations:
- `cm_rf.png` — Random Forest
- `cm_lr.png` — Logistic Regression

### Real-Student Test Results
Real-response predictions (distribution + per-student confidence) are stored in `ml/saved_models/metrics.json`
and displayed in the `/test_results` page. The test size is dynamic based on the latest response file used.

---

## 🚀 API Endpoints

### Authentication (form-based routes)
```
GET  /register   → registration page
POST /register   → create user (full_name, username, email, password)

GET  /login      → login page
POST /login      → login (username, password)

GET  /logout     → clear session and redirect
```

### Survey & Prediction
```
GET  /survey
  Output: Weekly survey page (auto-selects next week 1–4)

POST /submit_survey
  Input: survey form fields (numeric sliders + categorical selects)
  Output: stores response + prediction + SHAP + recommendation, then redirects to /result/<week>

GET  /result/<week>
  Output: risk level + confidence + SHAP explanations + grouped charts + personalised plan

GET  /dashboard
  Output: week progress + trend + charts + latest recommendation

GET  /history
  Output: all-week comparison + overall trend summary

GET  /test_results
  Output: evaluation plots + real-student prediction distribution + per-student predictions

GET  /model_insights
  Output: model selection rationale + SHAP method + feature groups + risk profiles
```

---

## 🔮 Future Enhancements

### Short-Term (Weeks 1-2)
- [ ] Multi-language support (Hindi, Kannada, Tamil)
- [ ] Session improvements (remember me, password reset)
- [ ] Email verification for registration

### Mid-Term (Months 2-3)
- [ ] Counselor collaboration portal
- [ ] Mental health resource integration
- [ ] Mobile app (React Native/Flutter)
- [ ] Gamification (badges, leaderboards)
- [ ] Push notifications for survey reminders

### Long-Term (6+ Months)
- [ ] LSTM for temporal pattern recognition
- [ ] Wearable device integration (smartwatch data)
- [ ] Institutional admin dashboard
- [ ] Anonymous peer comparison
- [ ] Intervention impact tracking
- [ ] Multi-model ensemble predictions
- [ ] Kubernetes deployment & auto-scaling

---

## 🤝 Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

**Guidelines:**
- Follow PEP 8 Python style guide
- Add docstrings to functions
- Include unit tests for new features
- Update README.md if adding new features

---

## 📝 License

This project is licensed under the **MIT License** — see LICENSE file for details.

---

## 👤 Author & Contact

**Student:** Nandini  
**Institution:** RV University, Bangalore  
**Project Year:** 2024-25  
**Project Code:** BTech CSE Final Year (4 Credits)

**Questions or Feedback?**
- 📧 Email: [Your Email]
- 🔗 GitHub: [Your GitHub Profile]
- 💼 LinkedIn: [Your LinkedIn]

---

## 🙏 Acknowledgments

- **SHAP Theory:** Lundberg & Lee (2017) — "A Unified Approach to Interpreting Model Predictions"
- **Dataset:** GenZ Mental Wellness Synthetic Dataset
- **Framework:** Flask, scikit-learn, pandas, matplotlib
- **Mentors:** RV University faculty and project advisors

---

## 📚 References

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
2. Maslach, C., & Leiter, M. P. (2016). Understanding the burnout experience. *World Psychiatry*, 15(2), 103-111.
3. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.

---

**Last Updated:** April 2025  
**Status:** ✅ Complete & Production-Ready
4. Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830.
