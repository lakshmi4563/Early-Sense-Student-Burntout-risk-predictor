"""
email_service.py
────────────────
Complete email notification system for BurnoutGuard.
Uses Python's built-in smtplib + Gmail SMTP (App Password).
No external packages needed.

Notifications sent:
  1. Welcome email       → on registration
  2. Week 1 reminder     → 1 day after registration (if not submitted)
  3. Weekly reminder     → every 7 days (Week 2, 3, 4 prompt)
  4. Prediction result   → immediately after each survey submission
  5. Trend alert         → if risk worsens (Improving→Declining or Low→High)
  6. Completion          → after all 4 weeks done

Setup (add to .env or config):
  MAIL_SENDER  = your_gmail@gmail.com
  MAIL_PASSWORD = your_16_char_app_password   (Gmail → Security → App Passwords)
"""

import smtplib
import threading
import time
import sqlite3
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
load_dotenv()

# ── Config (set these in your environment or hardcode for demo) ──
MAIL_SENDER   = os.environ.get('BURNOUT_MAIL_SENDER',   'burnoutguard.noreply@gmail.com')
MAIL_PASSWORD = os.environ.get('BURNOUT_MAIL_PASSWORD', '')   # Set your Gmail App Password
MAIL_ENABLED  = bool(MAIL_PASSWORD)                           # Auto-disabled if no password set
DB_PATH       = os.path.join(os.path.dirname(__file__), 'database', 'burnout.db')

PALETTE = {'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c'}
RISK_EMOJI = {'Low': '✅', 'Medium': '🟡', 'High': '⚠️'}

# ══════════════════════════════════════════════════════════
# BASE EMAIL TEMPLATE (dark branded HTML)
# ══════════════════════════════════════════════════════════
def _base_email(title: str, body_html: str, cta_text: str = '', cta_url: str = '') -> str:
    cta_block = ''
    if cta_text and cta_url:
        cta_block = f'''
        <div style="text-align:center;margin:32px 0;">
          <a href="{cta_url}"
             style="display:inline-block;background:#6c63ff;color:white;
                    text-decoration:none;padding:14px 32px;border-radius:10px;
                    font-weight:700;font-size:15px;letter-spacing:0.02em;">
            {cta_text}
          </a>
        </div>'''
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#080812;font-family:'Segoe UI',Arial,sans-serif;color:#e8e8f0;">
  <div style="max-width:580px;margin:0 auto;padding:32px 16px;">

    <!-- Header -->
    <div style="text-align:center;margin-bottom:28px;">
      <div style="display:inline-flex;align-items:center;gap:8px;">
        <div style="width:12px;height:12px;background:#6c63ff;border-radius:50%;
                    box-shadow:0 0 10px #6c63ff;"></div>
        <span style="font-size:18px;font-weight:700;letter-spacing:0.05em;">BurnoutGuard</span>
      </div>
      <div style="font-size:11px;color:#4a4a6a;margin-top:4px;letter-spacing:0.1em;text-transform:uppercase;">
        Student Wellness Tracking System · RV University
      </div>
    </div>

    <!-- Card -->
    <div style="background:#0f0f23;border:1px solid #2a2a4a;border-radius:16px;padding:32px;">
      <h1 style="font-size:22px;font-weight:700;margin:0 0 8px;color:#e8e8f0;">{title}</h1>
      <div style="height:2px;background:linear-gradient(90deg,#6c63ff,transparent);
                  border-radius:2px;margin-bottom:24px;"></div>
      {body_html}
      {cta_block}
    </div>

    <!-- Footer -->
    <div style="text-align:center;margin-top:24px;font-size:11px;color:#4a4a6a;line-height:1.7;">
      BurnoutGuard — Nandini · BTech CSE Final Year · RV University 2024–25<br>
      This is an automated notification. <a href="http://localhost:5000" style="color:#6c63ff;">Visit Dashboard</a>
    </div>
  </div>
</body>
</html>"""


def _p(text):
    return f'<p style="font-size:15px;color:#8888aa;line-height:1.7;margin:0 0 16px;">{text}</p>'

def _highlight(text, color='#6c63ff'):
    return f'<span style="color:{color};font-weight:700;">{text}</span>'

def _card(label, value, color='#6c63ff'):
    return f'''<div style="background:#15152e;border:1px solid #2a2a4a;border-radius:10px;
                           padding:14px 18px;margin-bottom:10px;display:flex;
                           justify-content:space-between;align-items:center;">
      <span style="font-size:13px;color:#8888aa;">{label}</span>
      <span style="font-size:15px;font-weight:700;color:{color};font-family:monospace;">{value}</span>
    </div>'''

def _tip_list(tips):
    items = ''.join(f'<li style="margin-bottom:8px;color:#8888aa;font-size:14px;">{t}</li>' for t in tips)
    return f'<ul style="padding-left:20px;margin:0 0 16px;">{items}</ul>'


# ══════════════════════════════════════════════════════════
# EMAIL SENDER
# ══════════════════════════════════════════════════════════
def send_email(to_email: str, subject: str, html_body: str) -> bool:
    """Send HTML email via Gmail SMTP. Returns True on success."""
    if not MAIL_ENABLED:
        # Log to console when email is not configured (for demo/testing)
        print(f'[EMAIL DISABLED] To: {to_email} | Subject: {subject}')
        return False
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = f'BurnoutGuard <{MAIL_SENDER}>'
        msg['To']      = to_email
        msg.attach(MIMEText(html_body, 'html', 'utf-8'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(MAIL_SENDER, MAIL_PASSWORD)
            server.sendmail(MAIL_SENDER, to_email, msg.as_string())
        print(f'[EMAIL SENT] To: {to_email} | {subject}')
        return True
    except Exception as e:
        print(f'[EMAIL ERROR] {e}')
        return False


# ══════════════════════════════════════════════════════════
# INDIVIDUAL NOTIFICATION FUNCTIONS
# ══════════════════════════════════════════════════════════

def send_welcome_email(user_name: str, email: str, username: str):
    """Sent immediately on registration."""
    body = f"""
    {_p(f"Hi <strong style='color:#e8e8f0;'>{user_name}</strong> 👋")}
    {_p("Welcome to <strong>BurnoutGuard</strong> — your 4-week burnout risk tracking system. You're now enrolled in the programme.")}
    {_p("Here's how it works:")}
    <ol style="color:#8888aa;font-size:14px;line-height:2;padding-left:20px;margin:0 0 20px;">
      <li>Fill the <strong style='color:#e8e8f0;'>Week 1 Survey</strong> (3 minutes, 19 questions)</li>
      <li>Get your <strong style='color:#6c63ff;'>AI prediction</strong> with SHAP explanations</li>
      <li>Follow the <strong style='color:#e8e8f0;'>personalised recovery plan</strong></li>
      <li>Return each week to track your trend over 4 weeks</li>
    </ol>
    {_p(f"Your username is <strong style='color:#6c63ff;'>{username}</strong>. Your Week 1 survey is ready now.")}
    {_p("We'll send you a reminder if you haven't submitted each week. You can unsubscribe anytime from the dashboard.")}
    """
    html = _base_email('Welcome to BurnoutGuard! 🎯', body,
                       'Start Week 1 Survey →', 'http://localhost:5000/survey')
    return send_email(email, 'Welcome to BurnoutGuard — Start Your Week 1 Survey', html)


def send_survey_reminder(user_name: str, email: str, week_number: int, days_overdue: int = 0):
    """Reminder to fill the weekly survey."""
    urgency = '⏰ Reminder' if days_overdue < 3 else '❗ Action Required'
    body = f"""
    {_p(f"Hi <strong style='color:#e8e8f0;'>{user_name}</strong>,")}
    {_p(f"Your <strong>Week {week_number} Burnout Risk Survey</strong> is ready and waiting for you.")}
    {_card('Current Week', f'Week {week_number} of 4', '#6c63ff')}
    {_card('Survey Time', '~3 minutes', '#00d4aa')}
    {_card('Questions', '19 wellness questions', '#f39c12')}
    {_p("Regular tracking is what makes the system powerful. Each submission helps the AI understand your trend — <strong style='color:#e8e8f0;'>Improving, Declining, Stable, or Fluctuating</strong>.")}
    {_p("Don't break your streak! 🔥")}
    """
    html = _base_email(f'{urgency}: Week {week_number} Survey', body,
                       f'Fill Week {week_number} Survey →', 'http://localhost:5000/survey')
    return send_email(email, f'[BurnoutGuard] Week {week_number} Survey — Fill Now', html)


def send_prediction_result(user_name: str, email: str, week_number: int,
                           risk_level: str, confidence: dict, top_features: list,
                           recommendation_summary: str):
    """Sent immediately after survey submission with prediction result."""
    risk_color = PALETTE.get(risk_level, '#888')
    risk_emoji = RISK_EMOJI.get(risk_level, '📊')
    top_feat_html = ''
    for i, feat in enumerate(top_features[:3], 1):
        arrow = '↑' if feat['direction'] == 'increases' else '↓'
        color = '#e74c3c' if feat['direction'] == 'increases' else '#27ae60'
        top_feat_html += _card(f'#{i} {feat["name"]}',
                               f'{arrow} SHAP: {feat["shap_value"]:+.4f}', color)
    body = f"""
    {_p(f"Hi <strong style='color:#e8e8f0;'>{user_name}</strong>,")}
    {_p(f"Your <strong>Week {week_number}</strong> burnout risk has been predicted:")}
    <div style="text-align:center;padding:24px;background:#15152e;border-radius:12px;
                border:2px solid {risk_color}33;margin-bottom:20px;">
      <div style="font-size:48px;font-weight:900;color:{risk_color};">{risk_level}</div>
      <div style="font-size:13px;color:#8888aa;margin-top:8px;">Burnout Risk Level</div>
      <div style="font-size:13px;margin-top:12px;color:#8888aa;">
        Confidence: Low {confidence.get('Low',0)*100:.0f}% &nbsp;|&nbsp;
        Medium {confidence.get('Medium',0)*100:.0f}% &nbsp;|&nbsp;
        High {confidence.get('High',0)*100:.0f}%
      </div>
    </div>
    <p style="font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;
              color:#8888aa;margin:0 0 10px;">Top Influencing Factors (SHAP)</p>
    {top_feat_html}
    {_p(f'<em>{recommendation_summary}</em>')}
    {_p("See your full SHAP explanation, grouped analysis, and personalised recovery plan on the dashboard.")}
    """
    html = _base_email(f'{risk_emoji} Week {week_number} Result: {risk_level} Risk', body,
                       'View Full Analysis →', f'http://localhost:5000/result/{week_number}')
    return send_email(email, f'[BurnoutGuard] Week {week_number}: {risk_level} Burnout Risk Predicted', html)


def send_trend_alert(user_name: str, email: str, trend: str, details: str,
                     current_risk: str, prev_risk: str, week_number: int):
    """Sent when trend changes significantly — especially if worsening."""
    if 'Declining' in trend or 'High' in current_risk:
        alert_type = '⚠️ Risk Increasing — Action Needed'
        alert_color = '#e74c3c'
        action_msg = 'Your burnout risk has increased. Please review the recommendations immediately.'
    elif 'Improving' in trend:
        alert_type = '📈 Great Progress!'
        alert_color = '#27ae60'
        action_msg = 'Your burnout risk is decreasing. Keep up the healthy habits!'
    else:
        return False   # Don't send for stable/fluctuating
    body = f"""
    {_p(f"Hi <strong style='color:#e8e8f0;'>{user_name}</strong>,")}
    {_p(f"We detected a <strong style='color:{alert_color};'>change in your burnout trend</strong> after Week {week_number}:")}
    {_card('Previous Risk', prev_risk, PALETTE.get(prev_risk,'#888'))}
    {_card('Current Risk', current_risk, PALETTE.get(current_risk,'#888'))}
    {_card('Trend', trend, alert_color)}
    {_p(f'<strong style="color:#e8e8f0;">{action_msg}</strong>')}
    {_p(details)}
    """
    html = _base_email(alert_type, body,
                       'View Dashboard →', 'http://localhost:5000/dashboard')
    return send_email(email, f'[BurnoutGuard] Trend Alert — {trend}', html)


def send_completion_email(user_name: str, email: str, risk_sequence: list,
                          final_trend: str, final_risk: str):
    """Sent after all 4 weeks are completed."""
    sequence_html = ''
    for i, risk in enumerate(risk_sequence, 1):
        sequence_html += _card(f'Week {i}', risk, PALETTE.get(risk, '#888'))
    body = f"""
    {_p(f"Hi <strong style='color:#e8e8f0;'>{user_name}</strong>,")}
    {_p("🎉 Congratulations! You have completed all <strong>4 weeks</strong> of the BurnoutGuard tracking programme.")}
    <p style="font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;
              color:#8888aa;margin:0 0 10px;">Your Journey</p>
    {sequence_html}
    {_card('Overall Trend', final_trend, '#6c63ff')}
    {_card('Final Risk Level', final_risk, PALETTE.get(final_risk,'#888'))}
    {_p("View your complete history, SHAP grouped analysis, and the full comparison table on the History page.")}
    {_p("Thank you for participating in this research. Your data contributes to understanding student wellness at RV University. 💙")}
    """
    html = _base_email('🎓 4-Week Programme Complete!', body,
                       'View Full History →', 'http://localhost:5000/history')
    return send_email(email, '[BurnoutGuard] 🎓 All 4 Weeks Complete — Final Report', html)


# ══════════════════════════════════════════════════════════
# BACKGROUND SCHEDULER (runs in a daemon thread)
# Checks every hour for users who need reminders
# ══════════════════════════════════════════════════════════
class NotificationScheduler:
    """
    Background thread that runs every CHECK_INTERVAL_HOURS hours.
    Sends reminders to users who registered but haven't submitted
    the expected week's survey.

    Logic:
      - User registered > 24 hrs ago AND week 1 not submitted → send Week 1 reminder
      - User submitted week N > 7 days ago AND week N+1 not submitted → send reminder
      - Only sends one reminder per (user, week) per day (throttled via last_notified table)
    """
    CHECK_INTERVAL_HOURS = 6   # Check every 6 hours

    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()
        self._init_notification_table()

    def _init_notification_table(self):
        """Add notification_log table to DB if not present."""
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notification_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER NOT NULL,
                    notif_type  TEXT NOT NULL,
                    week_number INTEGER,
                    sent_at     TEXT DEFAULT (datetime('now')),
                    success     INTEGER DEFAULT 1
                )""")
            conn.commit(); conn.close()
        except Exception as e:
            print(f'[NOTIFY] DB init error: {e}')

    def _already_sent_today(self, user_id: int, notif_type: str, week: int) -> bool:
        """Prevent duplicate sends within 24 hours."""
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """SELECT id FROM notification_log
                   WHERE user_id=? AND notif_type=? AND week_number=?
                   AND sent_at >= datetime('now', '-24 hours')""",
                (user_id, notif_type, week)
            ).fetchone()
            conn.close()
            return row is not None
        except:
            return False

    def _log(self, user_id: int, notif_type: str, week: int, success: bool):
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute('INSERT INTO notification_log(user_id,notif_type,week_number,success) VALUES(?,?,?,?)',
                         (user_id, notif_type, week, int(success)))
            conn.commit(); conn.close()
        except:
            pass

    def _check_and_send(self):
        """Main check loop — runs every CHECK_INTERVAL_HOURS."""
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            users = conn.execute('SELECT * FROM users').fetchall()
            now = datetime.utcnow()

            for user in users:
                uid   = user['id']
                name  = user['full_name']
                email = user['email']
                reg_dt = datetime.strptime(user['created_at'][:19], '%Y-%m-%d %H:%M:%S')

                # Get submitted weeks
                weeks_done = [r['week_number'] for r in
                              conn.execute('SELECT week_number FROM weekly_responses WHERE user_id=? ORDER BY week_number', (uid,)).fetchall()]
                next_week = max(weeks_done) + 1 if weeks_done else 1

                if next_week > 4:
                    continue   # All done

                # Week 1 reminder: if registered > 24h ago and no submission
                if not weeks_done:
                    hours_since_reg = (now - reg_dt).total_seconds() / 3600
                    if hours_since_reg >= 24 and not self._already_sent_today(uid, 'reminder', 1):
                        days_overdue = int(hours_since_reg / 24)
                        ok = send_survey_reminder(name, email, 1, days_overdue)
                        self._log(uid, 'reminder', 1, ok)
                else:
                    # Weekly reminder: if last submission was >7 days ago
                    last_resp = conn.execute(
                        'SELECT submitted_at FROM weekly_responses WHERE user_id=? ORDER BY week_number DESC LIMIT 1', (uid,)
                    ).fetchone()
                    if last_resp:
                        sub_dt = datetime.strptime(last_resp['submitted_at'][:19], '%Y-%m-%d %H:%M:%S')
                        days_since = (now - sub_dt).days
                        if days_since >= 7 and not self._already_sent_today(uid, 'reminder', next_week):
                            ok = send_survey_reminder(name, email, next_week, days_since - 7)
                            self._log(uid, 'reminder', next_week, ok)

            conn.close()
        except Exception as e:
            print(f'[NOTIFY SCHEDULER] Error: {e}')

    def _run(self):
        print(f'[NOTIFY] Scheduler started — checking every {self.CHECK_INTERVAL_HOURS}h')
        while not self._stop_event.is_set():
            self._check_and_send()
            # Sleep in small increments so stop_event is responsive
            for _ in range(self.CHECK_INTERVAL_HOURS * 360):  # 360 × 10s = 1h
                if self._stop_event.is_set():
                    break
                time.sleep(10)

    def start(self):
        if not self._thread or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True, name='NotifScheduler')
            self._thread.start()

    def stop(self):
        self._stop_event.set()


# Global scheduler instance
scheduler = NotificationScheduler()
