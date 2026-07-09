import json
from datetime import datetime

status = {
    'last_run': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'sentinel2_date': '2026-05-23',
    'images_used': 24,
    'ndvi_mean': 0.2456,
    'ndbi_mean': 0.1075,
    'alerts': [],
    'status': 'SUCCESS'
}

with open('../data/last_update.json', 'w') as f:
    json.dump(status, f, indent=2)

print('Timestamp saved.')