# ============================================
# DSS - KHAMANON BLOCK
# Script 13: Weather API Integration
# Source: OpenWeatherMap API (Free Tier)
# Location: Khamanon, Fatehgarh Sahib
# ============================================

import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from datetime import datetime

print("=" * 55)
print("  PRIORITY 3 — WEATHER INTEGRATION")
print("  Khamanon Block, Fatehgarh Sahib")
print("=" * 55)

base = os.path.dirname(os.path.abspath(__file__))

# ============================================
# CONFIGURATION
# Replace YOUR_API_KEY_HERE with your key
# ============================================

API_KEY  = "aad8b57ba2b85482fa8be2bbbed79489"
LAT      = 30.795   # Khamanon Block centre
LON      = 76.352
CITY     = "Khamanon, Fatehgarh Sahib"
UNITS    = "metric"  # Celsius

# ============================================
# STEP 1: CURRENT WEATHER
# ============================================

print(f"\nFetching current weather for {CITY}...")

current_url = (
    f"https://api.openweathermap.org/data/2.5/weather"
    f"?lat={LAT}&lon={LON}"
    f"&appid={API_KEY}&units={UNITS}"
)

try:
    response = requests.get(current_url, timeout=10)
    response.raise_for_status()
    current  = response.json()

    weather = {
        'timestamp'      : datetime.now().strftime(
                            '%Y-%m-%d %H:%M:%S'),
        'temperature'    : current['main']['temp'],
        'feels_like'     : current['main']['feels_like'],
        'humidity'       : current['main']['humidity'],
        'pressure'       : current['main']['pressure'],
        'description'    : current['weather'][0]['description'],
        'wind_speed'     : current['wind']['speed'],
        'wind_direction' : current['wind'].get('deg', 0),
        'cloudiness'     : current['clouds']['all'],
        'visibility'     : current.get('visibility', 0) / 1000,
        'rain_1h'        : current.get('rain', {}).get('1h', 0),
        'location'       : CITY
    }

    print(f"\nCurrent Weather — {CITY}:")
    print(f"  Temperature  : {weather['temperature']}°C "
          f"(feels like {weather['feels_like']}°C)")
    print(f"  Humidity     : {weather['humidity']}%")
    print(f"  Condition    : {weather['description'].title()}")
    print(f"  Wind         : {weather['wind_speed']} m/s")
    print(f"  Cloudiness   : {weather['cloudiness']}%")
    print(f"  Rainfall 1hr : {weather['rain_1h']} mm")

except Exception as e:
    print(f"Weather API error: {e}")
    print("Using fallback values for testing...")
    weather = {
        'timestamp'      : datetime.now().strftime(
                            '%Y-%m-%d %H:%M:%S'),
        'temperature'    : 35.0,
        'feels_like'     : 38.0,
        'humidity'       : 45,
        'pressure'       : 1005,
        'description'    : 'haze',
        'wind_speed'     : 3.2,
        'wind_direction' : 180,
        'cloudiness'     : 20,
        'visibility'     : 5.0,
        'rain_1h'        : 0,
        'location'       : CITY
    }

# ============================================
# STEP 2: 5-DAY FORECAST
# ============================================

print(f"\nFetching 5-day forecast...")

forecast_url = (
    f"https://api.openweathermap.org/data/2.5/forecast"
    f"?lat={LAT}&lon={LON}"
    f"&appid={API_KEY}&units={UNITS}"
)

forecast_data = []

try:
    response  = requests.get(forecast_url, timeout=10)
    response.raise_for_status()
    forecast  = response.json()

    for item in forecast['list']:
        forecast_data.append({
            'datetime'   : datetime.fromtimestamp(
                            item['dt']
                           ).strftime('%Y-%m-%d %H:%M'),
            'date'       : datetime.fromtimestamp(
                            item['dt']
                           ).strftime('%Y-%m-%d'),
            'temp'       : item['main']['temp'],
            'humidity'   : item['main']['humidity'],
            'description': item['weather'][0]['description'],
            'rain'       : item.get('rain', {}).get('3h', 0),
            'wind_speed' : item['wind']['speed'],
            'cloudiness' : item['clouds']['all']
        })

    print(f"Forecast points received: {len(forecast_data)}")

except Exception as e:
    print(f"Forecast API error: {e}")
    print("Using simulated forecast...")
    from datetime import timedelta
    base_date = datetime.now()
    for i in range(40):
        dt = base_date + timedelta(hours=i*3)
        forecast_data.append({
            'datetime'   : dt.strftime('%Y-%m-%d %H:%M'),
            'date'       : dt.strftime('%Y-%m-%d'),
            'temp'       : 35 + (i % 5),
            'humidity'   : 45 + (i % 20),
            'description': 'partly cloudy',
            'rain'       : 0,
            'wind_speed' : 3.0,
            'cloudiness' : 30
        })

forecast_df = pd.DataFrame(forecast_data)

# Daily summary
daily = forecast_df.groupby('date').agg({
    'temp'    : ['mean', 'max', 'min'],
    'humidity': 'mean',
    'rain'    : 'sum',
    'cloudiness': 'mean'
}).round(1)

print(f"\n5-Day Daily Summary:")
print(daily.to_string())

# ============================================
# STEP 3: WEATHER-BASED CROP ADVISORIES
# Link weather to crop stress interpretation
# ============================================

print("\nGenerating weather-based advisories...")

weather_advisories = []

# Temperature stress
if weather['temperature'] > 40:
    weather_advisories.append({
        'type'    : 'Heat Stress',
        'severity': 'CRITICAL',
        'message' : f"🔴 Extreme heat: {weather['temperature']}°C. "
                    f"Avoid field operations. "
                    f"Irrigate in early morning or evening."
    })
elif weather['temperature'] > 35:
    weather_advisories.append({
        'type'    : 'Heat Stress',
        'severity': 'WARNING',
        'message' : f"⚠️ High temperature: {weather['temperature']}°C. "
                    f"Monitor crop for heat stress. "
                    f"Ensure adequate soil moisture."
    })

# Humidity
if weather['humidity'] < 30:
    weather_advisories.append({
        'type'    : 'Low Humidity',
        'severity': 'WARNING',
        'message' : f"⚠️ Low humidity: {weather['humidity']}%. "
                    f"High evapotranspiration. "
                    f"Check soil moisture daily."
    })
elif weather['humidity'] > 80:
    weather_advisories.append({
        'type'    : 'High Humidity',
        'severity': 'WARNING',
        'message' : f"⚠️ High humidity: {weather['humidity']}%. "
                    f"Fungal disease risk elevated. "
                    f"Monitor for blast, blight, rust."
    })

# Rainfall
if weather['rain_1h'] > 20:
    weather_advisories.append({
        'type'    : 'Heavy Rainfall',
        'severity': 'WARNING',
        'message' : f"⚠️ Heavy rainfall: {weather['rain_1h']} mm/hr. "
                    f"Risk of waterlogging and nutrient leaching. "
                    f"Check field drainage."
    })
elif weather['rain_1h'] == 0 and weather['humidity'] < 40:
    weather_advisories.append({
        'type'    : 'Drought Risk',
        'severity': 'INFO',
        'message' : f"ℹ️ No recent rainfall. "
                    f"Monitor soil moisture. "
                    f"Consider irrigation if crops show stress."
    })

# Wind
if weather['wind_speed'] > 10:
    weather_advisories.append({
        'type'    : 'High Wind',
        'severity': 'WARNING',
        'message' : f"⚠️ High wind: {weather['wind_speed']} m/s. "
                    f"Avoid pesticide/fertilizer application. "
                    f"Risk of lodging in tall crops."
    })

# Cloudiness effect on satellite
if weather['cloudiness'] > 70:
    weather_advisories.append({
        'type'    : 'Cloud Cover',
        'severity': 'INFO',
        'message' : f"ℹ️ High cloud cover: {weather['cloudiness']}%. "
                    f"May affect Sentinel-2 image quality. "
                    f"Next clear acquisition may be delayed."
    })

if not weather_advisories:
    weather_advisories.append({
        'type'    : 'General',
        'severity': 'OK',
        'message' : '✅ Weather conditions are favourable for field operations.'
    })

print(f"\nWeather advisories generated: {len(weather_advisories)}")
for a in weather_advisories:
    print(f"  [{a['severity']}] {a['type']}: {a['message'][:60]}...")

# ============================================
# PLOT 1: Weather Dashboard
# Temperature + Humidity + Rain forecast
# ============================================

print("\nGenerating weather charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

dates      = forecast_df['date'].unique()[:5]
date_temps = [
    forecast_df[forecast_df['date']==d]['temp'].mean()
    for d in dates
]
date_rain  = [
    forecast_df[forecast_df['date']==d]['rain'].sum()
    for d in dates
]
date_hum   = [
    forecast_df[forecast_df['date']==d]['humidity'].mean()
    for d in dates
]
date_cloud = [
    forecast_df[forecast_df['date']==d]['cloudiness'].mean()
    for d in dates
]

# Temperature trend
colors_temp = [
    '#e74c3c' if t > 35
    else '#f39c12' if t > 30
    else '#2ecc71'
    for t in date_temps
]
axes[0,0].bar(
    range(len(dates)), date_temps,
    color=colors_temp, alpha=0.85,
    edgecolor='black', linewidth=0.5
)
axes[0,0].axhline(
    y=35, color='red',
    linestyle='--', linewidth=1.5,
    label='Heat stress (35°C)'
)
axes[0,0].set_xticks(range(len(dates)))
axes[0,0].set_xticklabels(
    [d[5:] for d in dates],
    fontsize=9
)
axes[0,0].set_ylabel('Temperature (°C)', fontsize=10)
axes[0,0].set_title(
    '5-Day Temperature Forecast',
    fontsize=11, fontweight='bold'
)
axes[0,0].legend(fontsize=8)
axes[0,0].grid(True, alpha=0.3, axis='y')

# Rainfall
axes[0,1].bar(
    range(len(dates)), date_rain,
    color='#3498db', alpha=0.85,
    edgecolor='black', linewidth=0.5
)
axes[0,1].set_xticks(range(len(dates)))
axes[0,1].set_xticklabels(
    [d[5:] for d in dates], fontsize=9
)
axes[0,1].set_ylabel('Rainfall (mm)', fontsize=10)
axes[0,1].set_title(
    '5-Day Rainfall Forecast',
    fontsize=11, fontweight='bold'
)
axes[0,1].grid(True, alpha=0.3, axis='y')

# Humidity
axes[1,0].plot(
    range(len(dates)), date_hum,
    'g-o', linewidth=2.5, markersize=8
)
axes[1,0].axhline(
    y=80, color='red',
    linestyle='--', linewidth=1.5,
    label='Disease risk (80%)'
)
axes[1,0].axhline(
    y=30, color='orange',
    linestyle='--', linewidth=1.5,
    label='Drought risk (30%)'
)
axes[1,0].set_xticks(range(len(dates)))
axes[1,0].set_xticklabels(
    [d[5:] for d in dates], fontsize=9
)
axes[1,0].set_ylabel('Humidity (%)', fontsize=10)
axes[1,0].set_title(
    '5-Day Humidity Forecast',
    fontsize=11, fontweight='bold'
)
axes[1,0].legend(fontsize=8)
axes[1,0].grid(True, alpha=0.3)

# Cloud cover
axes[1,1].bar(
    range(len(dates)), date_cloud,
    color='#95a5a6', alpha=0.85,
    edgecolor='black', linewidth=0.5
)
axes[1,1].axhline(
    y=70, color='blue',
    linestyle='--', linewidth=1.5,
    label='S2 quality affected (70%)'
)
axes[1,1].set_xticks(range(len(dates)))
axes[1,1].set_xticklabels(
    [d[5:] for d in dates], fontsize=9
)
axes[1,1].set_ylabel('Cloud Cover (%)', fontsize=10)
axes[1,1].set_title(
    '5-Day Cloud Cover\n(affects Sentinel-2 quality)',
    fontsize=11, fontweight='bold'
)
axes[1,1].legend(fontsize=8)
axes[1,1].grid(True, alpha=0.3, axis='y')

plt.suptitle(
    f'Weather Forecast — {CITY}\n'
    f'Current: {weather["temperature"]}°C | '
    f'{weather["description"].title()} | '
    f'Humidity: {weather["humidity"]}%',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join(base, '..', 'maps',
                 'weather_forecast.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/weather_forecast.png")

# ============================================
# SAVE WEATHER DATA
# ============================================

weather_path = os.path.join(
    base, '..', 'data', 'current_weather.json'
)
weather['advisories'] = weather_advisories

with open(weather_path, 'w') as f:
    json.dump(weather, f, indent=2)

forecast_df.to_csv(
    os.path.join(base, '..', 'data',
                 'weather_forecast.csv'),
    index=False
)

print("Saved: data/current_weather.json")
print("Saved: data/weather_forecast.csv")

print("\n" + "=" * 55)
print("  PRIORITY 3 COMPLETE")
print("=" * 55)
print(f"\nCurrent weather for {CITY}:")
print(f"  Temperature  : {weather['temperature']}°C")
print(f"  Humidity     : {weather['humidity']}%")
print(f"  Condition    : {weather['description'].title()}")
print(f"\nWeather advisories: {len(weather_advisories)}")
for a in weather_advisories:
    print(f"  [{a['severity']}] {a['message'][:70]}")
print("\nNext: Priority 4 — Risk Assessment Maps")