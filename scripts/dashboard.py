"""Servidor local del dashboard en tiempo real.

    python scripts/dashboard.py            # abre http://localhost:8765
    python scripts/train_relative_v2.py    # (en otra terminal) el dashboard se actualiza solo

Sirve dashboard/index.html y los ficheros de progreso de reports/ (training_progress.json, live_feed.json).
"""
from __future__ import annotations

import http.server
import socketserver
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKELETON = ('<!doctype html><html lang="es"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>{}</body></html>')


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            body = SKELETON.replace("{}", (ROOT / "dashboard" / "index.html").read_text()).encode()
            return self._send(body, "text/html; charset=utf-8")
        f = ROOT / "reports" / path.lstrip("/")
        if path.endswith(".json") and f.exists():
            return self._send(f.read_bytes(), "application/json")
        self.send_error(404)

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as srv:
        url = f"http://localhost:{port}"
        print(f"Dashboard en {url} (Ctrl+C para salir)")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        srv.serve_forever()
