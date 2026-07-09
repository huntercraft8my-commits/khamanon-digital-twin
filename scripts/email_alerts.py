# ============================================
# DSS - KHAMANON BLOCK
# Script 15: Email Alert System
# ============================================

import smtplib
import json
import pandas as pd
import os
from datetime import datetime
from email.mime.text      import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base      import MIMEBase
from email                import encoders

print("=" * 55)
print("  PRIORITY 5 — EMAIL ALERT SYSTEM")
print("  Khamanon Block DSS")
print("=" * 55)

base = os.path.dirname(os.path.abspath(__file__))

# ============================================
# CONFIGURATION — replace with your values
# ============================================

SENDER_EMAIL    = "huntercraft8my@gmail.com"
SENDER_PASSWORD = "YOUR_16_CHAR_APP_PASSWORD"
RECEIVER_EMAIL  = "huntercraft8my@gmail.com"

# ============================================
# LOAD DATA
# ============================================

print("\nLoading system data...")

status_path = os.path.join(base,'..','data','last_update.json')
if os.path.exists(status_path):
    with open(status_path,'r') as f:
        status = json.load(f)
else:
    status = {'last_run':'Unknown','sentinel2_date':'Unknown',
               'ndvi_mean':0.35,'alerts':[]}

weather_path = os.path.join(base,'..','data','current_weather.json')
if os.path.exists(weather_path):
    with open(weather_path,'r') as f:
        weather = json.load(f)
else:
    weather = {'temperature':35,'humidity':45,
               'description':'unavailable','advisories':[],
               'wind_speed':3,'rain_1h':0}

adv_path = os.path.join(base,'..','data','block_advisories.csv')
block_adv = pd.read_csv(adv_path) if os.path.exists(adv_path) else pd.DataFrame()

risk_path = os.path.join(base,'..','data','point_risk_scores.csv')
risk_df   = pd.read_csv(risk_path) if os.path.exists(risk_path) else pd.DataFrame()

ndvi_mean = status.get('ndvi_mean', 0.35)
s2_date   = status.get('sentinel2_date', 'Unknown')
last_run  = status.get('last_run', 'Unknown')
w_temp    = weather.get('temperature', 35)
w_hum     = weather.get('humidity', 45)
w_desc    = str(weather.get('description','N/A')).title()
w_wind    = weather.get('wind_speed', 0)
w_rain    = weather.get('rain_1h', 0)

print(f"  NDVI      : {ndvi_mean:.3f}")
print(f"  S2 date   : {s2_date}")
print(f"  Temp      : {w_temp}C")

# ============================================
# DETERMINE ALERT LEVEL
# ============================================

alert_level   = 'ROUTINE'
alert_color   = '#27ae60'
critical_items= []
warning_items = []

if ndvi_mean < 0.20:
    alert_level = 'CRITICAL'
    alert_color = '#e74c3c'
    critical_items.append('Severe crop stress: NDVI=' + str(round(ndvi_mean,3)))
elif ndvi_mean < 0.35:
    alert_level = 'WARNING'
    alert_color = '#e67e22'
    warning_items.append('Crop stress: NDVI=' + str(round(ndvi_mean,3)))

if w_temp > 40:
    alert_level = 'CRITICAL'
    alert_color = '#e74c3c'
    critical_items.append('Extreme heat: ' + str(w_temp) + 'C')
elif w_temp > 35:
    if alert_level == 'ROUTINE':
        alert_level = 'WARNING'
        alert_color = '#e67e22'
    warning_items.append('High temperature: ' + str(w_temp) + 'C')

print(f"\nAlert level: {alert_level}")

# ============================================
# BUILD EMAIL
# ============================================

now_str = datetime.now().strftime('%d %B %Y, %H:%M IST')

sev_colors = {
    'CRITICAL':'#e74c3c','WARNING':'#e67e22',
    'INFO':'#3498db','OK':'#27ae60'
}

# Build advisory rows
adv_rows = ''
if not block_adv.empty:
    for _, row in block_adv.iterrows():
        c = sev_colors.get(str(row.get('severity','INFO')),'#3498db')
        rid = str(row.get('rule_id',''))
        msg = str(row.get('message',''))
        act = str(row.get('action',''))
        adv_rows += (
            '<tr><td style="padding:8px;border-bottom:1px solid #ecf0f1;'
            'color:' + c + ';font-weight:bold;">[' + rid + '] ' + msg + '</td></tr>'
            '<tr><td style="padding:4px 8px 12px 20px;border-bottom:1px solid #ecf0f1;'
            'color:#555;font-size:13px;">' + act + '</td></tr>'
        )

if not adv_rows:
    adv_rows = '<tr><td style="padding:10px;color:#27ae60;">All parameters within normal range.</td></tr>'

# Build risk section
risk_section = ''
if not risk_df.empty:
    dm = round(risk_df['degradation_risk'].mean())
    cm = round(risk_df['crop_failure_risk'].mean())
    sm = round(risk_df['salinity_risk'].mean())
    om = round(risk_df['overall_risk'].mean())

    def rc(s):
        if s > 70: return '#e74c3c'
        if s > 40: return '#e67e22'
        return '#27ae60'

    risk_section = (
        '<h3 style="color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:5px;">Risk Assessment</h3>'
        '<table width="100%" cellpadding="0" cellspacing="8"><tr>'
        '<td style="text-align:center;background:#f8f9fa;padding:12px;border-radius:8px;">'
        '<div style="font-size:26px;font-weight:bold;color:' + rc(dm) + ';">' + str(dm) + '</div>'
        '<div style="color:#7f8c8d;font-size:11px;">Degradation /100</div></td>'
        '<td style="text-align:center;background:#f8f9fa;padding:12px;border-radius:8px;">'
        '<div style="font-size:26px;font-weight:bold;color:' + rc(cm) + ';">' + str(cm) + '</div>'
        '<div style="color:#7f8c8d;font-size:11px;">Crop Failure /100</div></td>'
        '<td style="text-align:center;background:#f8f9fa;padding:12px;border-radius:8px;">'
        '<div style="font-size:26px;font-weight:bold;color:' + rc(sm) + ';">' + str(sm) + '</div>'
        '<div style="color:#7f8c8d;font-size:11px;">Salinity /100</div></td>'
        '<td style="text-align:center;background:#f8f9fa;padding:12px;border-radius:8px;">'
        '<div style="font-size:26px;font-weight:bold;color:' + rc(om) + ';">' + str(om) + '</div>'
        '<div style="color:#7f8c8d;font-size:11px;">Overall /100</div></td>'
        '</tr></table><br>'
    )

# Build alert blocks
critical_html = ''
for item in critical_items:
    critical_html += (
        '<div style="background:#fdf2f2;border-left:4px solid #e74c3c;'
        'padding:12px 15px;margin:8px 0;border-radius:4px;">'
        '<strong style="color:#e74c3c;">CRITICAL: ' + item + '</strong></div>'
    )

warning_html = ''
for item in warning_items:
    warning_html += (
        '<div style="background:#fef9e7;border-left:4px solid #e67e22;'
        'padding:12px 15px;margin:8px 0;border-radius:4px;">'
        '<strong style="color:#e67e22;">WARNING: ' + item + '</strong></div>'
    )

ndvi_col = '#e74c3c' if ndvi_mean < 0.35 else '#27ae60'

# Final HTML
html = (
    '<!DOCTYPE html><html><head><meta charset="UTF-8"></head><body '
    'style="font-family:Segoe UI,Arial,sans-serif;background:#f0f2f5;margin:0;padding:20px;">'
    '<div style="max-width:700px;margin:0 auto;background:white;border-radius:12px;'
    'overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.1);">'

    # Header
    '<div style="background:' + alert_color + ';padding:25px 30px;">'
    '<h1 style="color:white;margin:0;font-size:22px;">Khamanon Block DSS</h1>'
    '<p style="color:rgba(255,255,255,0.85);margin:5px 0 0 0;font-size:13px;">'
    'Fatehgarh Sahib, Punjab | Alert Level: <strong>' + alert_level + '</strong>'
    ' | ' + now_str + '</p></div>'

    # Body
    '<div style="padding:25px 30px;">'

    # Status
    '<h3 style="color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:5px;">System Status</h3>'
    '<table width="100%" style="border-collapse:collapse;">'
    '<tr><td style="padding:6px;color:#7f8c8d;width:40%;">Last Update Run</td>'
    '<td style="padding:6px;font-weight:bold;">' + last_run + '</td></tr>'
    '<tr style="background:#f8f9fa;"><td style="padding:6px;color:#7f8c8d;">Sentinel-2 Date</td>'
    '<td style="padding:6px;font-weight:bold;">' + s2_date + '</td></tr>'
    '<tr><td style="padding:6px;color:#7f8c8d;">NDVI (Block Mean)</td>'
    '<td style="padding:6px;font-weight:bold;color:' + ndvi_col + ';">' + str(round(ndvi_mean,3)) + '</td></tr>'
    '<tr style="background:#f8f9fa;"><td style="padding:6px;color:#7f8c8d;">Live Dashboard</td>'
    '<td style="padding:6px;"><a href="https://khamanon-dss.onrender.com" '
    'style="color:#3498db;">khamanon-dss.onrender.com</a></td></tr>'
    '</table><br>'

    # Weather
    '<h3 style="color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:5px;">Current Weather</h3>'
    '<table width="100%" style="border-collapse:collapse;"><tr>'
    '<td style="padding:8px;background:#fff3cd;border-radius:6px;text-align:center;width:25%;">'
    '<strong style="font-size:24px;">' + str(w_temp) + 'C</strong><br><small>Temperature</small></td>'
    '<td style="padding:8px;text-align:center;width:25%;">'
    '<strong>' + str(w_hum) + '%</strong><br><small>Humidity</small></td>'
    '<td style="padding:8px;text-align:center;width:25%;">'
    '<strong>' + str(w_wind) + ' m/s</strong><br><small>Wind</small></td>'
    '<td style="padding:8px;text-align:center;width:25%;">'
    '<strong>' + str(w_rain) + ' mm</strong><br><small>Rainfall 1hr</small></td>'
    '</tr></table>'
    '<p style="color:#7f8c8d;font-size:12px;margin:5px 0;">Condition: ' + w_desc + '</p><br>'

    # Risk
    + risk_section +

    # PAU Advisories
    '<h3 style="color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:5px;">'
    'PAU Soil Advisory (Source: PAU Package of Practices 2025)</h3>'
    '<table width="100%" style="border-collapse:collapse;">' + adv_rows + '</table><br>'

    # Alerts
    + critical_html + warning_html +

    '</div>'

    # Footer
    '<div style="background:#2c3e50;padding:15px 30px;text-align:center;">'
    '<p style="color:#bdc3c7;margin:0;font-size:12px;">'
    'Khamanon Block DSS | M.Sc. Research, PAU Ludhiana | Fatehgarh Sahib, Punjab</p>'
    '<p style="color:#7f8c8d;margin:5px 0 0 0;font-size:11px;">'
    'Automated alert from DSS. Data: Sentinel-2 + PAU 2025.</p>'
    '</div></div></body></html>'
)

# ============================================
# SEND EMAIL
# ============================================

def send_email():
    print("\nPreparing email...")

    subject = (
        '[Khamanon DSS] '
        + alert_level + ' Alert | '
        'NDVI:' + str(round(ndvi_mean,3))
        + ' | Temp:' + str(w_temp) + 'C'
        + ' | ' + datetime.now().strftime('%d %b %Y')
    )

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = SENDER_EMAIL
    msg['To']      = RECEIVER_EMAIL
    msg.attach(MIMEText(html, 'html'))

    adv_attach = os.path.join(base,'..','data','block_advisories.csv')
    if os.path.exists(adv_attach):
        with open(adv_attach,'rb') as f:
            part = MIMEBase('application','octet-stream')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition',
                            'attachment; filename=block_advisories.csv')
            msg.attach(part)

    print("Connecting to Gmail SMTP...")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print("Email sent successfully!")
        print("  To     : " + RECEIVER_EMAIL)
        print("  Subject: " + subject)
        return True
    except smtplib.SMTPAuthenticationError:
        print("Authentication failed. Check App Password.")
        return False
    except Exception as e:
        print("Email error: " + str(e))
        return False

success = send_email()

print("\n" + "=" * 55)
print("  PRIORITY 5 COMPLETE")
print("=" * 55)
if success:
    print("\nCheck inbox: " + RECEIVER_EMAIL)
else:
    print("\nEmail not sent — check App Password.")
print("\nNext: Priority 6 — Auto Scheduler")