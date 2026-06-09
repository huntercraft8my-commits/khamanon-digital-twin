# ============================================================
# Entry point for Render / gunicorn deployment
# ============================================================
import sys
import os

# Add scripts/ to the import path so dashboard_v2 is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from dashboard_v2 import app          # noqa: E402
server = app.server                   # gunicorn target: app:server

if __name__ == '__main__':
    app.run(
        debug=False,
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 8050))
    )
