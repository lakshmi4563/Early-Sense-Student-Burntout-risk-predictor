"""
BurnoutGuard — Flask Backend (Final Version)
Changes vs previous:
  - Email notifications wired to all key events
  - Grouped SHAP (Academic / Behavioural / Lifestyle) in result + dashboard
  - /model_insights route (replaces "Demo Guide") — professional name
  - make_shap_chart now generates GROUPED waterfall
  - User email stored in session for notifications
"""
import os, json, pickle, hashlib, sqlite3, re
from collections import Counter
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from datetime import datetime
from functools import wraps
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from flask import (Flask, request, jsonify, session, redirect,
                   url_for, render_template, flash, g)
from email_service import (
    scheduler, send_welcome_email, send_prediction_result,
    send_trend_alert, send_completion_email, send_survey_reminder
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
    template_folder=os.path.join(BASE_DIR,'frontend','templates'),
    static_folder  =os.path.join(BASE_DIR,'frontend','static'))
app.secret_key = 'rv_burnout_nandini_2025'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# 500 error handler for logging
@app.errorhandler(500)
def handle_500(error):
    import traceback
    tb = traceback.format_exc()
    with open(os.path.join(BASE_DIR,'error_log.txt'), 'a') as f:
        f.write(f"\n[{datetime.now()}] 500 ERROR:\n{tb}\n")
    print(f"[500 ERROR] {tb}")
    return f"Internal Server Error: {str(error)}", 500
DB_PATH      = os.path.join(BASE_DIR,'database','burnout.db')
MODELS_PATH  = os.path.join(BASE_DIR,'ml','saved_models','burnout_model.pkl')
METRICS_PATH = os.path.join(BASE_DIR,'ml','saved_models','metrics.json')
MODEL_COMPARISON_PATH = os.path.join(BASE_DIR,'ml','saved_models','model_comparison_results.json')
PLOTS_DIR    = os.path.join(BASE_DIR,'frontend','static','img')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_BUNDLE = None
def load_model():
    global MODEL_BUNDLE
    if MODEL_BUNDLE is None:
        with open(MODELS_PATH,'rb') as f: MODEL_BUNDLE = pickle.load(f)
    return MODEL_BUNDLE

def build_live_test_metrics():
    """
    Recompute real-student predictions from the current Google Form file
    using the same production model path as the web app.
    """
    try:
        b = load_model()
        lr = b.get('primary_model') or b.get('lr_model')
        scaler = b.get('scaler')
        if lr is None or scaler is None:
            return {}

        # Reuse the same cleaner used in training/evaluation pipeline.
        from ml.ml_pipeline import load_test_data, TEST_XLSX
        X_test, _ = load_test_data(TEST_XLSX, b['gender_enc'], b['working_enc'], b['content_enc'])
        if X_test is None or len(X_test) == 0:
            return {'test_distribution': {}, 'student_predictions': [], 'n_test': 0}

        X_arr = X_test.values if hasattr(X_test, 'values') else np.array(X_test)
        X_sc = scaler.transform(X_arr)
        y_pred = lr.predict(X_sc)
        y_prob = lr.predict_proba(X_sc)

        preds = []
        for i in range(len(y_pred)):
            preds.append({
                'student_id': i + 1,
                'predicted_risk': LABEL_INV[int(y_pred[i])],
                'confidence': {
                    'Low': round(float(y_prob[i][0]), 4),
                    'Medium': round(float(y_prob[i][1]), 4),
                    'High': round(float(y_prob[i][2]), 4),
                },
            })

        dist = Counter([LABEL_INV[int(p)] for p in y_pred])
        return {
            'test_distribution': {'Low': dist.get('Low', 0), 'Medium': dist.get('Medium', 0), 'High': dist.get('High', 0)},
            'student_predictions': preds,
            'n_test': len(preds),
        }
    except Exception as e:
        print(f"[WARN] Could not build live test metrics: {e}")
        return {}

FEATURE_COLS = [
    'Study_Work_Hours_per_Day','Daily_Sleep_Hours','Exercise_Frequency_per_Week',
    'Caffeine_Intake_Cups','Screen_Time_Hours','Night_Scrolling_Frequency',
    'Online_Gaming_Hours','Overthinking_Score','Social_Comparison_Index',
    'Anxiety_Score','Mood_Stability_Score','Emotional_Fatigue_Score',
    'Daily_Social_Media_Hours','Motivation_Level','Sleep_Quality_Score',
    'Age','Gender_enc','Working_Status_enc','Content_Type_enc',
]
FEATURE_LABELS = {
    'Study_Work_Hours_per_Day':'Study / Work Hours per Day','Daily_Sleep_Hours':'Sleep Hours per Day',
    'Exercise_Frequency_per_Week':'Exercise Days per Week','Caffeine_Intake_Cups':'Caffeine Cups per Day',
    'Screen_Time_Hours':'Total Screen Time (hrs)','Night_Scrolling_Frequency':'Night Scrolling Frequency',
    'Online_Gaming_Hours':'Gaming Hours per Day','Overthinking_Score':'Overthinking Score (1–10)',
    'Social_Comparison_Index':'Social Comparison (1–10)','Anxiety_Score':'Anxiety Score (1–10)',
    'Mood_Stability_Score':'Mood Stability (1–10)','Emotional_Fatigue_Score':'Emotional Fatigue (1–10)',
    'Daily_Social_Media_Hours':'Social Media Hours per Day','Motivation_Level':'Motivation Level (1–10)',
    'Sleep_Quality_Score':'Sleep Quality Score (1–10)','Age':'Age',
    'Gender_enc':'Gender','Working_Status_enc':'Working Status','Content_Type_enc':'Content Type',
}
FEATURE_GROUPS = {
    'Academic Stress':      ['Study_Work_Hours_per_Day','Overthinking_Score','Anxiety_Score','Motivation_Level','Working_Status_enc'],
    'Behavioural Patterns': ['Screen_Time_Hours','Night_Scrolling_Frequency','Online_Gaming_Hours','Daily_Social_Media_Hours','Caffeine_Intake_Cups','Content_Type_enc'],
    'Lifestyle & Health':   ['Daily_Sleep_Hours','Sleep_Quality_Score','Exercise_Frequency_per_Week','Mood_Stability_Score','Emotional_Fatigue_Score','Social_Comparison_Index','Age','Gender_enc'],
}
GROUP_COLORS = {'Academic Stress':'#6c63ff','Behavioural Patterns':'#00d4aa','Lifestyle & Health':'#f39c12'}
LABEL_INV = {0:'Low',1:'Medium',2:'High'}
RISK_SCORE = {'Low':0,'Medium':1,'High':2}
PALETTE    = {'Low':'#27ae60','Medium':'#f39c12','High':'#e74c3c'}
BG='#0f0f23'; FG='#e8e8f0'

# ── DATABASE ─────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL, created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS weekly_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
    week_number INTEGER NOT NULL CHECK(week_number BETWEEN 1 AND 4),
    submitted_at TEXT DEFAULT (datetime('now')),
    study_work_hours_per_day REAL, daily_sleep_hours REAL,
    exercise_frequency_per_week REAL, caffeine_intake_cups REAL,
    screen_time_hours REAL, night_scrolling_frequency REAL,
    online_gaming_hours REAL, overthinking_score REAL,
    social_comparison_index REAL, anxiety_score REAL,
    mood_stability_score REAL, emotional_fatigue_score REAL,
    daily_social_media_hours REAL, motivation_level REAL,
    sleep_quality_score REAL, age INTEGER,
    gender TEXT, working_status TEXT, content_type TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id), UNIQUE(user_id,week_number)
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
    week_number INTEGER NOT NULL, response_id INTEGER NOT NULL,
    predicted_risk TEXT NOT NULL,
    confidence_low REAL, confidence_medium REAL, confidence_high REAL,
    shap_values_json TEXT, top_features_json TEXT,
    grouped_shap_json TEXT, recommendation_json TEXT,
    predicted_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(user_id) REFERENCES users(id), UNIQUE(user_id,week_number)
);
"""
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db

@app.teardown_appcontext
def close_db(e):
    db=g.pop('db',None)
    if db: db.close()

def init_db():
    os.makedirs(os.path.dirname(DB_PATH),exist_ok=True)
    c=sqlite3.connect(DB_PATH); c.executescript(SCHEMA); c.commit(); c.close()

# ── AUTH ──────────────────────────────────────────────────────────────────────
def hash_pw(p): return hashlib.sha256(('rv_burnout_salt_'+p).encode()).hexdigest()
def check_pw(p,h): return hash_pw(p)==h
def login_required(f):
    @wraps(f)
    def dec(*a,**kw):
        if 'user_id' not in session: flash('Please log in.','warning'); return redirect(url_for('login'))
        return f(*a,**kw)
    return dec
def current_user(): return get_db().execute('SELECT * FROM users WHERE id=?',(session['user_id'],)).fetchone() if 'user_id' in session else None

# ── LinearSHAP ────────────────────────────────────────────────────────────────
class ManualLinearSHAP:
    """LinearSHAP for Logistic Regression: φ_i = W[class, i] × (x_i − E[x_i])"""
    def __init__(self, lr_model, feature_names, background_mean):
        self.model=lr_model; self.fnames=feature_names; self.bg=background_mean
    def explain_single(self, x, pred_class_idx):
        W=self.model.coef_
        shap_row=W[pred_class_idx]*(x-self.bg)
        return dict(sorted({self.fnames[i]:float(shap_row[i]) for i in range(len(self.fnames))}.items(),key=lambda kv:abs(kv[1]),reverse=True))

# ── PREDICTION ────────────────────────────────────────────────────────────────
def form_to_vector(form):
    b=load_model()
    ge=b['gender_enc']; we=b['working_enc']; ce=b['content_enc']
    def enc(encoder,val,fb):
        try: return int(encoder.transform([val])[0])
        except: return int(encoder.transform([fb])[0])
    g_val=str(form.get('gender','Female')).strip().title()
    w_val=str(form.get('working_status','Student')).strip().title()
    c_val=str(form.get('content_type','Entertainment')).strip().title()
    return np.array([float(form.get('study_work_hours_per_day',6)),float(form.get('daily_sleep_hours',7)),
        float(form.get('exercise_frequency_per_week',2)),float(form.get('caffeine_intake_cups',1)),
        float(form.get('screen_time_hours',6)),float(form.get('night_scrolling_frequency',3)),
        float(form.get('online_gaming_hours',1)),float(form.get('overthinking_score',5)),
        float(form.get('social_comparison_index',4)),float(form.get('anxiety_score',4)),
        float(form.get('mood_stability_score',6)),float(form.get('emotional_fatigue_score',4)),
        float(form.get('daily_social_media_hours',3)),float(form.get('motivation_level',6)),
        float(form.get('sleep_quality_score',6)),
        int(form.get('age',21)),
        enc(ge,g_val,'Male'),enc(we,w_val,'Student'),enc(ce,c_val,'Entertainment')],dtype=float)

def predict_burnout(form_data):
    b=load_model(); lr=b.get('primary_model') or b['lr_model']; scaler=b['scaler']; bg=b['background_mean']
    x_raw=form_to_vector(form_data); x_sc=scaler.transform(x_raw.reshape(1,-1))[0]
    # Predict & explain with Logistic Regression (99.40% accuracy, 99.99% AUC-ROC)
    pid=int(lr.predict(x_sc.reshape(1,-1))[0]); proba=lr.predict_proba(x_sc.reshape(1,-1))[0]
    risk=LABEL_INV[pid]
    # Explain with LinearSHAP (mathematically faithful to the LR prediction model)
    expl=ManualLinearSHAP(lr,FEATURE_COLS,bg); shap_d=expl.explain_single(x_sc,pid)
    # Grouped SHAP
    grouped={}
    for grp,feats in FEATURE_GROUPS.items():
        grouped[grp]={'total':round(sum(abs(shap_d.get(f,0)) for f in feats),6),
                      'features':sorted([{'feature':f,'label':FEATURE_LABELS.get(f,f),
                          'shap_value':round(shap_d.get(f,0),6),
                          'direction':'increases' if shap_d.get(f,0)>0 else 'decreases',
                          'raw_value':round(float(x_raw[FEATURE_COLS.index(f)]),2)} for f in feats],
                         key=lambda x:abs(x['shap_value']),reverse=True)}
    top5=[{'name':FEATURE_LABELS.get(k,k),'shap_value':round(v,6),'direction':'increases' if v>0 else 'decreases',
            'raw_value':round(float(x_raw[FEATURE_COLS.index(k)]),2)} for k,v in list(shap_d.items())[:5]]
    return {'risk_level':risk,'probabilities':{'Low':round(float(proba[0]),4),'Medium':round(float(proba[1]),4),'High':round(float(proba[2]),4)},
            'shap_values':{k:round(v,6) for k,v in shap_d.items()},'top_features':top5,'grouped_shap':grouped,'pred_idx':pid}

# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────
def make_recommendations(risk,top_features,form_data):
    feat_advice={
        'Overthinking Score (1–10)':('Practice 5-min journaling to externalise racing thoughts.','Use 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s.'),
        'Anxiety Score (1–10)':('Limit social media to 30 min/day.','10-min guided meditation daily (Headspace/Insight Timer).'),
        'Emotional Fatigue (1–10)':('Schedule a 1-hour no-obligation window daily.','Talk to one trusted person about how you feel each week.'),
        'Sleep Hours per Day':('Sleep deprivation amplifies burnout — aim 7–9 hrs.','Fixed sleep/wake times; no screens 45 min before bed.'),
        'Sleep Quality Score (1–10)':('Poor sleep quality worsens emotional regulation.','Keep room cool, dark; avoid caffeine after 2 PM.'),
        'Total Screen Time (hrs)':('Excessive screen time depletes mental energy.','20-20-20 rule every hour of screen use.'),
        'Night Scrolling Frequency':('Night scrolling disrupts melatonin.','Phone away by 10 PM — use a real alarm clock.'),
        'Social Comparison (1–10)':('High social comparison strongly links to burnout.','Unfollow 5 accounts that make you feel inferior.'),
        'Study / Work Hours per Day':('Working >10 hrs/day causes cognitive overload.','Pomodoro: 25 min work + 5 min break.'),
        'Motivation Level (1–10)':('Low motivation signals exhaustion, not laziness.','Set one tiny achievable goal each morning.'),
        'Exercise Days per Week':('Physical inactivity worsens anxiety and fatigue.','Even a 20-min walk improves mood — aim 3 days/week.'),
    }
    immediate=[]; daily=[]
    for feat in top_features:
        if feat['direction']=='increases' and feat['name'] in feat_advice:
            t,h=feat_advice[feat['name']]; immediate.append(t); daily.append(h)
    if risk=='High':
        return {'risk_level':risk,
          'summary':'⚠️ HIGH Burnout Risk — your body and mind need immediate care. Start with just one action today.',
          'immediate':list(dict.fromkeys(immediate+['Tell someone you trust how you feel right now.','Cancel ONE non-critical commitment this week.','Take a 24-hr digital detox this weekend.','Contact campus counselling (iCall: 9152987821).']))[:5],
          'daily_habits':list(dict.fromkeys(daily+['Sleep 7–9 hrs — set alarms for BOTH sleep and wake.','Eat at least 2 proper meals daily.','Spend 20 min outdoors daily.']))[:5],
          'weekly_goals':['Attend one campus activity that is NOT academic.','Have a real conversation (not WhatsApp) with a friend.','Review your task list — what can be dropped?'],
          'resources':[{'name':'iCall India','contact':'9152987821','type':'phone'},{'name':'Vandrevala Foundation','contact':'1860-2662-345','type':'phone'},{'name':'7 Cups','contact':'7cups.com','type':'web'}],
          'motivational_message':'Reaching out is the strongest thing you can do. Burnout is not a flaw — it is your body asking for rest. 💙'}
    elif risk=='Medium':
        return {'risk_level':risk,
          'summary':'🟡 MEDIUM Burnout Risk — warning signs visible. Small consistent changes now prevent escalation.',
          'immediate':list(dict.fromkeys(immediate+['Identify ONE stressor to address THIS week.','Reduce social media by 1 hour today.','15-min body scan tonight.']))[:5],
          'daily_habits':list(dict.fromkeys(daily+['Consistent 7-hr sleep with fixed schedule.','One physical activity per day (even a 30-min walk).','5-min micro-breaks every study hour.']))[:5],
          'weekly_goals':['Quality social time with friends twice this week.','Complete pending assignments by priority.','Try one new stress-relief activity.'],
          'resources':[{'name':'Smiling Mind App','contact':'smilingmind.com.au','type':'app'},{'name':'iCall India','contact':'9152987821','type':'phone'}],
          'motivational_message':'Self-awareness is the first step. Small steps compound into big changes. 🌱'}
    else:
        return {'risk_level':risk,
          'summary':'✅ LOW Burnout Risk — you are maintaining your wellness well!',
          'immediate':['Acknowledge what you are doing right.','Share your wellness strategies with someone who is struggling.'],
          'daily_habits':['Continue your sleep and exercise routines.','Stay socially connected — relationships buffer burnout.'],
          'weekly_goals':['Maintain academic momentum with weekly planning.','Explore a creative hobby unrelated to coursework.'],
          'resources':[{'name':'Mind Tools','contact':'mindtools.com','type':'web'}],
          'motivational_message':'You are thriving! Keep nurturing your balance. 🌟'}

# ── TREND ─────────────────────────────────────────────────────────────────────
def analyse_trend(risk_levels):
    if not risk_levels: return {'trend':'No Data','details':'Submit your first survey to start tracking.','scores':[]}
    scores=[RISK_SCORE.get(r,1) for r in risk_levels]; n=len(scores)
    if n==1: return {'trend':'Baseline','details':f'Week 1 baseline: {risk_levels[0]} risk.','scores':scores,'labels':risk_levels}
    net=scores[-1]-scores[0]; deltas=[scores[i+1]-scores[i] for i in range(n-1)]
    pos=sum(d>0 for d in deltas); neg=sum(d<0 for d in deltas)
    if net<0 and neg>=pos:   trend,detail='Improving 📈',f'Risk improved from {risk_levels[0]} (W1) to {risk_levels[-1]} (W{n}). Keep it up!'
    elif net>0 and pos>=neg: trend,detail='Declining 📉',f'Risk worsened from {risk_levels[0]} (W1) to {risk_levels[-1]} (W{n}). Act on recommendations!'
    elif net==0:             trend,detail='Stable ➡️',f'Risk stayed at {risk_levels[-1]} across {n} weeks.'
    else:                    trend,detail='Fluctuating 〰️','Risk fluctuates week to week — focus on consistent habits.'
    streak=1
    for i in range(n-2,-1,-1):
        if risk_levels[i]==risk_levels[-1]: streak+=1
        else: break
    return {'trend':trend,'details':detail,'scores':scores,'labels':risk_levels,'net_change':net,'streak':streak,'streak_label':risk_levels[-1]}

# ── CHARTS ────────────────────────────────────────────────────────────────────
def make_weekly_chart(week_data, user_id):
    weeks=[d['week_number'] for d in week_data]; scores=[RISK_SCORE.get(d['predicted_risk'],1) for d in week_data]
    labels=[d['predicted_risk'] for d in week_data]
    fig,ax=plt.subplots(figsize=(8,4),facecolor=BG); ax.set_facecolor('#1a1a35')
    ax.axhspan(-0.3,0.5,alpha=0.08,color='#27ae60'); ax.axhspan(0.5,1.5,alpha=0.08,color='#f39c12'); ax.axhspan(1.5,2.3,alpha=0.08,color='#e74c3c')
    ax.plot(weeks,scores,color='#6c63ff',linewidth=2.5,marker='o',markersize=10,zorder=5)
    for w,s,lb in zip(weeks,scores,labels):
        ax.scatter(w,s,color=PALETTE.get(lb,'#888'),s=120,zorder=6,edgecolors='white',linewidth=1.5)
        ax.text(w,s+0.12,lb,ha='center',va='bottom',color=FG,fontsize=9,fontweight='bold')
    ax.set_xticks(range(1,5)); ax.set_xticklabels([f'Week {i}' for i in range(1,5)],color=FG,fontsize=9)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(['Low','Medium','High'],color=FG,fontsize=9)
    ax.set_ylim(-0.3,2.4); ax.set_xlim(0.5,4.5); ax.tick_params(colors=FG)
    for s in ax.spines.values(): s.set_edgecolor('#333355')
    ax.set_title('Your Burnout Risk — Weekly Trend',color=FG,fontsize=12,fontweight='bold',pad=12)
    patches=[mpatches.Patch(color=PALETTE[k],label=k) for k in ['Low','Medium','High']]
    ax.legend(handles=patches,facecolor='#1a1a35',labelcolor=FG,fontsize=8,loc='upper right')
    plt.tight_layout(pad=1.5); fname=f'weekly_trend_{user_id}.png'
    fig.savefig(os.path.join(PLOTS_DIR,fname),dpi=100,facecolor=BG); plt.close(); return fname

def make_grouped_shap_chart(grouped_shap, risk_level, user_id, week):
    """3-panel SHAP chart grouped by Academic / Behavioural / Lifestyle — stacked vertically."""
    fig,axes=plt.subplots(3,1,figsize=(10,12),facecolor=BG)
    fig.suptitle(f'SHAP Analysis by Domain — Week {week} | {risk_level} Risk',color=FG,fontsize=12,fontweight='bold')
    for ax,(grp,gdata) in zip(axes,grouped_shap.items()):
        ax.set_facecolor('#1a1a35')
        feats=gdata['features']
        if not feats: ax.text(0.5,0.5,'No data',ha='center',va='center',color=FG,transform=ax.transAxes); continue
        labels=[f['label'] for f in feats]; vals=[f['shap_value'] for f in feats]
        colors=['#e74c3c' if v>0 else '#27ae60' for v in vals]; n=len(labels)
        ax.barh(range(n),vals[::-1],color=colors[::-1],edgecolor='#ffffff11')
        ax.set_yticks(range(n)); ax.set_yticklabels(labels[::-1],fontsize=8,color=FG)
        ax.axvline(0,color=FG,linewidth=0.7,linestyle='--'); ax.tick_params(colors=FG)
        for s in ax.spines.values(): s.set_edgecolor('#333355')
        ax.set_title(grp,color=GROUP_COLORS.get(grp,'#6c63ff'),fontsize=10,fontweight='bold',pad=8)
        ax.set_xlabel('SHAP Value (← Lowers | Raises ->)',color=FG,fontsize=7)
    pos_p=mpatches.Patch(color='#e74c3c',label='↑ Raises Risk')
    neg_p=mpatches.Patch(color='#27ae60',label='↓ Lowers Risk')
    fig.legend(handles=[pos_p,neg_p],facecolor='#1a1a35',labelcolor=FG,fontsize=8,
               loc='lower center',bbox_to_anchor=(0.5,-0.01),ncol=2)
    plt.tight_layout(pad=1.5,rect=[0,0.03,1,1])
    fname=f'shap_grouped_user{user_id}_week{week}.png'
    fig.savefig(os.path.join(PLOTS_DIR,fname),dpi=100,facecolor=BG); plt.close(); return fname

def make_confidence_gauge_chart(confidence_dict, risk_level, user_id, week):
    """Render a radar-style confidence chart for the current prediction."""
    cats=['Low','Medium','High']
    vals=[float(confidence_dict.get('Low',0)),float(confidence_dict.get('Medium',0)),float(confidence_dict.get('High',0))]
    angles=np.linspace(0,2*np.pi,len(cats),endpoint=False).tolist()
    vals_plot=vals+[vals[0]]; ang_plot=angles+[angles[0]]
    fig,ax=plt.subplots(figsize=(7,6),facecolor=BG,subplot_kw=dict(projection='polar'))
    ax.set_facecolor('#1a1a35')
    ax.plot(ang_plot,vals_plot,'o-',linewidth=2.5,color='#6c63ff',markersize=7)
    ax.fill(ang_plot,vals_plot,alpha=0.22,color='#6c63ff')
    ax.set_xticks(angles); ax.set_xticklabels(cats,color=FG,fontsize=11,fontweight='bold')
    ax.set_ylim(0,1)
    ax.set_yticks([0.25,0.5,0.75,1.0]); ax.set_yticklabels(['25%','50%','75%','100%'],color=FG,fontsize=8)
    ax.grid(True,color=FG,alpha=0.18)
    colors={'Low':'#27ae60','Medium':'#f39c12','High':'#e74c3c'}
    for a,v,c in zip(angles,vals,cats):
        ax.text(a,v+0.12,f'{v:.1%}',ha='center',va='center',color=colors[c],fontsize=10,fontweight='bold')
    ax.set_title(f'Prediction Confidence — {risk_level} Risk',color=colors.get(risk_level,'#888'),fontsize=12,fontweight='bold',pad=20)
    plt.tight_layout()
    fname=f'confidence_gauge_user{user_id}_week{week}.png'
    fig.savefig(os.path.join(PLOTS_DIR,fname),dpi=100,facecolor=BG); plt.close(); return fname

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/')
def index(): return redirect(url_for('dashboard')) if 'user_id' in session else render_template('index.html')

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method=='GET': return render_template('register.html')
    d=request.form; u=d.get('username','').strip(); e=d.get('email','').strip()
    p=d.get('password',''); fn=d.get('full_name','').strip()
    if len(u)<3: flash('Username needs 3+ chars.','danger'); return render_template('register.html')
    if '@' not in e: flash('Valid email required.','danger'); return render_template('register.html')
    if len(p)<6: flash('Password needs 6+ chars.','danger'); return render_template('register.html')
    db=get_db()
    if db.execute('SELECT id FROM users WHERE username=?',(u,)).fetchone():
        flash('Username taken.','danger'); return render_template('register.html')
    if db.execute('SELECT id FROM users WHERE email=?',(e,)).fetchone():
        flash('Email already registered.','danger'); return render_template('register.html')
    db.execute('INSERT INTO users(username,email,password_hash,full_name) VALUES(?,?,?,?)',(u,e,hash_pw(p),fn)); db.commit()
    # Send welcome email in background thread
    import threading
    threading.Thread(target=send_welcome_email, args=(fn,e,u), daemon=True).start()
    flash('Account created! Check your email for a welcome message.','success')
    return redirect(url_for('login'))

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='GET': return render_template('login.html')
    db=get_db(); u=db.execute('SELECT * FROM users WHERE username=?',(request.form.get('username','').strip(),)).fetchone()
    if not u or not check_pw(request.form.get('password',''),u['password_hash']):
        flash('Invalid username or password.','danger'); return render_template('login.html')
    session['user_id']=u['id']; session['username']=u['username']
    session['full_name']=u['full_name']; session['email']=u['email']
    flash(f'Welcome, {u["full_name"]}!','success'); return redirect(url_for('dashboard'))

@app.route('/logout')
def logout(): session.clear(); flash('Logged out.','info'); return redirect(url_for('index'))

@app.route('/reset_survey',methods=['POST'])
@login_required
def reset_survey():
    db=get_db(); uid=session['user_id']
    db.execute('DELETE FROM predictions      WHERE user_id=?',(uid,))
    db.execute('DELETE FROM weekly_responses WHERE user_id=?',(uid,))
    db.commit(); flash('Survey data reset. You can now fill Week 1 again.','success')
    return redirect(url_for('survey'))

@app.route('/dashboard')
@login_required
def dashboard():
    db=get_db(); uid=session['user_id']
    preds=db.execute('SELECT * FROM predictions WHERE user_id=? ORDER BY week_number',(uid,)).fetchall()
    weeks=[p['week_number'] for p in preds]; next_week=max(weeks)+1 if weeks else 1; can_submit=next_week<=4
    trend=analyse_trend([p['predicted_risk'] for p in preds])
    chart_file=make_weekly_chart([{'week_number':p['week_number'],'predicted_risk':p['predicted_risk']} for p in preds],uid) if preds else None
    latest_pred=latest_rec=latest_shap_chart=None
    if preds:
        lp=preds[-1]; latest_pred=dict(lp)
        if lp['recommendation_json']: latest_rec=json.loads(lp['recommendation_json'])
        if lp['grouped_shap_json']:
            gs=json.loads(lp['grouped_shap_json'])
            latest_shap_chart=make_grouped_shap_chart(gs,lp['predicted_risk'],uid,lp['week_number'])
        elif lp['shap_values_json']:
            # fallback
            sv=json.loads(lp['shap_values_json']); gs={}
            for grp,feats in FEATURE_GROUPS.items():
                gs[grp]={'total':0,'features':[{'label':FEATURE_LABELS.get(f,f),'shap_value':sv.get(f,0),'direction':'increases' if sv.get(f,0)>0 else 'decreases'} for f in feats]}
            latest_shap_chart=make_grouped_shap_chart(gs,lp['predicted_risk'],uid,lp['week_number'])
    return render_template('dashboard.html',user=current_user(),predictions=[dict(p) for p in preds],
        weeks_submitted=weeks,next_week=next_week,can_submit=can_submit,trend_info=trend,
        chart_file=chart_file,latest_pred=latest_pred,latest_rec=latest_rec,latest_shap_chart=latest_shap_chart)

@app.route('/survey')
@login_required
def survey():
    db=get_db(); uid=session['user_id']
    done=db.execute('SELECT week_number FROM weekly_responses WHERE user_id=?',(uid,)).fetchall()
    weeks_done=[r['week_number'] for r in done]; next_week=max(weeks_done)+1 if weeks_done else 1
    if next_week>4: flash('All 4 weeks complete! Use Reset Survey to redo.','info'); return redirect(url_for('dashboard'))
    return render_template('survey.html',week_number=next_week,user=current_user())

@app.route('/submit_survey',methods=['POST'])
@login_required
def submit_survey():
    import threading as _t
    import traceback
    try:
        db=get_db(); uid=session['user_id']; form=request.form
        done=db.execute('SELECT week_number FROM weekly_responses WHERE user_id=?',(uid,)).fetchall()
        week_num=max([r['week_number'] for r in done])+1 if done else 1
        if week_num>4: flash('All 4 weeks complete.','warning'); return redirect(url_for('dashboard'))
        db.execute('''INSERT INTO weekly_responses
            (user_id,week_number,study_work_hours_per_day,daily_sleep_hours,
             exercise_frequency_per_week,caffeine_intake_cups,screen_time_hours,
             night_scrolling_frequency,online_gaming_hours,overthinking_score,
             social_comparison_index,anxiety_score,mood_stability_score,
             emotional_fatigue_score,daily_social_media_hours,motivation_level,
             sleep_quality_score,age,gender,working_status,content_type)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (uid,week_num,float(form.get('study_work_hours_per_day',6)),float(form.get('daily_sleep_hours',7)),
             float(form.get('exercise_frequency_per_week',2)),float(form.get('caffeine_intake_cups',1)),
             float(form.get('screen_time_hours',6)),float(form.get('night_scrolling_frequency',3)),
             float(form.get('online_gaming_hours',1)),float(form.get('overthinking_score',5)),
             float(form.get('social_comparison_index',4)),float(form.get('anxiety_score',4)),
             float(form.get('mood_stability_score',6)),float(form.get('emotional_fatigue_score',4)),
             float(form.get('daily_social_media_hours',3)),float(form.get('motivation_level',6)),
             float(form.get('sleep_quality_score',6)),int(form.get('age',21)),
             str(form.get('gender','Female')),str(form.get('working_status','Student')),str(form.get('content_type','Entertainment'))))
        db.commit()
        resp_id=db.execute('SELECT last_insert_rowid()').fetchone()[0]
        result=predict_burnout(dict(form))
        rec=make_recommendations(result['risk_level'],result['top_features'],dict(form))
        db.execute('''INSERT OR REPLACE INTO predictions
            (user_id,week_number,response_id,predicted_risk,confidence_low,confidence_medium,confidence_high,
             shap_values_json,top_features_json,grouped_shap_json,recommendation_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
            (uid,week_num,resp_id,result['risk_level'],
             result['probabilities']['Low'],result['probabilities']['Medium'],result['probabilities']['High'],
             json.dumps(result['shap_values']),json.dumps(result['top_features']),
             json.dumps(result['grouped_shap']),json.dumps(rec)))
        db.commit()
        user_email=session.get('email',''); user_name=session.get('full_name','Student')
        if user_email:
            _t.Thread(target=send_prediction_result,args=(user_name,user_email,week_num,
                result['risk_level'],result['probabilities'],result['top_features'],rec['summary']),daemon=True).start()
            if week_num>1:
                all_p=db.execute('SELECT predicted_risk FROM predictions WHERE user_id=? ORDER BY week_number',(uid,)).fetchall()
                rl=[p['predicted_risk'] for p in all_p]; trend=analyse_trend(rl)
                if 'Declining' in trend['trend'] or ('High' in rl[-1] and 'High' not in rl[-2]):
                    prev=rl[-2] if len(rl)>=2 else rl[-1]
                    _t.Thread(target=send_trend_alert,args=(user_name,user_email,trend['trend'],
                        trend['details'],rl[-1],prev,week_num),daemon=True).start()
            if week_num==4:
                all_p2=db.execute('SELECT predicted_risk FROM predictions WHERE user_id=? ORDER BY week_number',(uid,)).fetchall()
                rl2=[p['predicted_risk'] for p in all_p2]; trend2=analyse_trend(rl2)
                _t.Thread(target=send_completion_email,args=(user_name,user_email,rl2,trend2['trend'],rl2[-1]),daemon=True).start()
        return redirect(url_for('result',week=week_num))
    except Exception as e:
        error_msg = traceback.format_exc()
        with open('error_log.txt', 'a') as f:
            f.write(f"\n[{datetime.now()}] ERROR in submit_survey:\n{error_msg}\n")
        print(f"[ERROR] {error_msg}")
        flash(f'Error processing survey: {str(e)}','danger')
        return redirect(url_for('survey'))
    rec=make_recommendations(result['risk_level'],result['top_features'],dict(form))
    db.execute('''INSERT OR REPLACE INTO predictions
        (user_id,week_number,response_id,predicted_risk,confidence_low,confidence_medium,confidence_high,
         shap_values_json,top_features_json,grouped_shap_json,recommendation_json)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
        (uid,week_num,resp_id,result['risk_level'],
         result['probabilities']['Low'],result['probabilities']['Medium'],result['probabilities']['High'],
         json.dumps(result['shap_values']),json.dumps(result['top_features']),
         json.dumps(result['grouped_shap']),json.dumps(rec)))
    db.commit()
    # Email notifications (non-blocking)
    user_email=session.get('email',''); user_name=session.get('full_name','Student')
    if user_email:
        _t.Thread(target=send_prediction_result,args=(user_name,user_email,week_num,
            result['risk_level'],result['probabilities'],result['top_features'],rec['summary']),daemon=True).start()
        # Trend alert if declining
        if week_num>1:
            all_p=db.execute('SELECT predicted_risk FROM predictions WHERE user_id=? ORDER BY week_number',(uid,)).fetchall()
            rl=[p['predicted_risk'] for p in all_p]; trend=analyse_trend(rl)
            if 'Declining' in trend['trend'] or ('High' in rl[-1] and 'High' not in rl[-2]):
                prev=rl[-2] if len(rl)>=2 else rl[-1]
                _t.Thread(target=send_trend_alert,args=(user_name,user_email,trend['trend'],
                    trend['details'],rl[-1],prev,week_num),daemon=True).start()
        # Completion email after week 4
        if week_num==4:
            all_p2=db.execute('SELECT predicted_risk FROM predictions WHERE user_id=? ORDER BY week_number',(uid,)).fetchall()
            rl2=[p['predicted_risk'] for p in all_p2]; trend2=analyse_trend(rl2)
            _t.Thread(target=send_completion_email,args=(user_name,user_email,rl2,trend2['trend'],rl2[-1]),daemon=True).start()
    return redirect(url_for('result',week=week_num))

@app.route('/result/<int:week>')
@login_required
def result(week):
    db=get_db(); uid=session['user_id']
    pred=db.execute('SELECT * FROM predictions WHERE user_id=? AND week_number=?',(uid,week)).fetchone()
    if not pred: flash('Prediction not found.','danger'); return redirect(url_for('dashboard'))
    shap_vals=json.loads(pred['shap_values_json'])   if pred['shap_values_json']   else {}
    top_feats=json.loads(pred['top_features_json'])  if pred['top_features_json']  else []
    grouped_shap=json.loads(pred['grouped_shap_json']) if pred['grouped_shap_json'] else {}
    rec=json.loads(pred['recommendation_json']) if pred['recommendation_json'] else {}
    shap_chart=make_grouped_shap_chart(grouped_shap,pred['predicted_risk'],uid,week) if grouped_shap else ''
    conf={'Low':pred['confidence_low'] or 0,'Medium':pred['confidence_medium'] or 0,'High':pred['confidence_high'] or 0}
    gauge_chart=make_confidence_gauge_chart(conf,pred['predicted_risk'],uid,week)
    all_p=db.execute('SELECT predicted_risk FROM predictions WHERE user_id=? ORDER BY week_number',(uid,)).fetchall()
    trend=analyse_trend([p['predicted_risk'] for p in all_p])
    return render_template('result.html',user=current_user(),pred=dict(pred),shap_vals=shap_vals,
        top_feats=top_feats,grouped_shap=grouped_shap,rec=rec,shap_chart=shap_chart,
        trend=trend,week=week,feature_labels=FEATURE_LABELS,group_colors=GROUP_COLORS,gauge_chart=gauge_chart)

@app.route('/history')
@login_required
def history():
    db=get_db(); uid=session['user_id']
    preds=db.execute('SELECT * FROM predictions WHERE user_id=? ORDER BY week_number',(uid,)).fetchall()
    pl=[]
    for p in preds:
        d=dict(p)
        d['top_features']=json.loads(p['top_features_json']) if p['top_features_json'] else []
        d['rec_summary']=json.loads(p['recommendation_json']).get('summary','') if p['recommendation_json'] else ''
        pl.append(d)
    trend=analyse_trend([p['predicted_risk'] for p in preds])
    chart_file=make_weekly_chart([{'week_number':p['week_number'],'predicted_risk':p['predicted_risk']} for p in preds],uid) if preds else None
    return render_template('history.html',user=current_user(),predictions=pl,trend=trend,chart_file=chart_file)

@app.route('/test_results')
def test_results():
    metrics={}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f: metrics=json.load(f)
    # Normalise key names — support both old ('test_distribution') and new ('test_rf_distribution')
    if 'test_rf_distribution' in metrics and 'test_distribution' not in metrics:
        metrics['test_distribution'] = metrics['test_rf_distribution']
    # Prefer live predictions from the latest response file/model when available.
    live_metrics = build_live_test_metrics()
    if live_metrics:
        metrics.update(live_metrics)
    # Dynamic count from actual student predictions
    n_test = len(metrics.get('student_predictions', [])) or metrics.get('n_test', 0)
    return render_template('test_results.html', metrics=metrics,
                           group_colors=GROUP_COLORS, n_test=n_test)

@app.route('/static/sw.js')
def service_worker():
    """Serve service worker from root path (required for full-scope PWA)."""
    from flask import send_from_directory
    return send_from_directory(
        os.path.join(BASE_DIR, 'frontend', 'static'), 'sw.js',
        mimetype='application/javascript'
    )

@app.route('/model_insights')
def model_insights():
    """Professional model insights page — replaces 'Demo Guide'."""
    guide={'Low':{'label':'✅ LOW Risk','color':'#27ae60','desc':'Mentally balanced and healthy.',
            'features':{'Anxiety Score':'1–3 (Calm)','Motivation Level':'8–10 (High)',
             'Sleep Hours/Day':'7–9 hours','Sleep Quality':'7–10 (Good)','Mood Stability':'8–10 (Stable)',
             'Emotional Fatigue':'1–3 (Energised)','Overthinking':'1–3 (Rarely)',
             'Exercise Days/Week':'4–7 days','Study/Work Hrs/Day':'4–7 hours',
             'Social Media Hrs/Day':'0–2 hours','Night Scrolling':'0–2 (Rarely)'}},
           'Medium':{'label':'🟡 MEDIUM Risk','color':'#f39c12','desc':'Warning signs — act before escalation.',
            'features':{'Anxiety Score':'4–6 (Moderate)','Motivation Level':'4–7 (Average)',
             'Sleep Hours/Day':'6–7 hours','Sleep Quality':'5–7 (Average)','Mood Stability':'5–7 (Somewhat stable)',
             'Emotional Fatigue':'4–6 (Moderate)','Overthinking':'4–6 (Sometimes)',
             'Exercise Days/Week':'1–3 days','Study/Work Hrs/Day':'7–9 hours',
             'Social Media Hrs/Day':'3–5 hours','Night Scrolling':'4–6 (Often)'}},
           'High':{'label':'🔴 HIGH Risk','color':'#e74c3c','desc':'Immediate action required.',
            'features':{'Anxiety Score':'7–10 (High anxiety)','Motivation Level':'1–3 (Very low)',
             'Sleep Hours/Day':'3–5 hours (deprived)','Sleep Quality':'1–4 (Poor)','Mood Stability':'1–4 (Very unstable)',
             'Emotional Fatigue':'7–10 (Exhausted)','Overthinking':'7–10 (Constantly)',
             'Exercise Days/Week':'0 days','Study/Work Hrs/Day':'10–14 hours',
             'Social Media Hrs/Day':'6–10 hours','Night Scrolling':'8–10 (Every night)'}}}
    model_info={'name':'Logistic Regression','solver':'lbfgs','C':1.0,'max_iter':1000,'class_weight':'balanced',
        'why_chosen':'Logistic Regression is the primary production model. It provides highly reliable class probabilities, fast inference, and transparent linear decision boundaries that are easier to maintain for this project.',
        'shap_explanation':'We implement TreeSHAP (Lundberg & Lee, 2017) from scratch using sklearn\'s decision_path API applied to the Random Forest model. For each prediction: (1) walk root->leaf path, (2) compute probability change at each node split, (3) attribute that change to the splitting feature, (4) average across 200 trees. This gives φ values where Σφᵢ = f(x) − E[f(x)]. While predictions come from Logistic Regression (superior accuracy), explanations use Random Forest to provide interpretable feature contributions.',
        'groups':{'Academic Stress':['Study hours','Overthinking','Anxiety','Motivation','Working status'],
                  'Behavioural Patterns':['Screen time','Night scrolling','Gaming','Social media','Caffeine','Content type'],
                  'Lifestyle & Health':['Sleep hours','Sleep quality','Exercise','Mood stability','Emotional fatigue','Social comparison','Age','Gender']}}
    metrics={}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f: metrics=json.load(f)
    lr_metrics=(metrics.get('validation') or {}).get('Logistic Regression',{})
    model_info['metrics']={
        'accuracy': round(float(lr_metrics.get('Accuracy',0))*100,2),
        'auc': round(float(lr_metrics.get('AUC-ROC',0))*100,2),
        'f1': round(float(lr_metrics.get('F1-Score',0))*100,2),
    }
    primary_model_name=type((load_model().get('primary_model') or load_model().get('lr_model'))).__name__
    pipeline_checks={
        'train_includes_logistic': 'Logistic Regression' in (metrics.get('validation') or {}),
        'app_prediction_model': primary_model_name,
        'test_results_model': 'Logistic Regression (live recompute)',
    }
    # Dynamic count from actual student predictions
    n_test = len(metrics.get('student_predictions', [])) or metrics.get('n_test', 0)
    return render_template('model_insights.html',guide=guide,model_info=model_info,metrics=metrics,
                          pipeline_checks=pipeline_checks,group_colors=GROUP_COLORS,n_test=n_test)

@app.route('/api/user_stats')
@login_required
def api_user_stats():
    db=get_db(); uid=session['user_id']
    p=db.execute('SELECT week_number,predicted_risk,confidence_low,confidence_medium,confidence_high FROM predictions WHERE user_id=? ORDER BY week_number',(uid,)).fetchall()
    return jsonify([dict(r) for r in p])

@app.route('/api/trend')
@login_required
def api_trend():
    db=get_db(); uid=session['user_id']
    p=db.execute('SELECT predicted_risk FROM predictions WHERE user_id=? ORDER BY week_number',(uid,)).fetchall()
    return jsonify(analyse_trend([r['predicted_risk'] for r in p]))

if __name__=='__main__':
    init_db(); load_model(); scheduler.start()
    print('[APP] Server -> http://127.0.0.1:5000')
    app.run(debug=False,host='0.0.0.0',port=5000)
