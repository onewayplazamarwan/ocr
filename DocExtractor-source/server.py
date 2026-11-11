"""
مستخرج بيانات الوثائق - سيرفر محلي
Local document-data extractor.

يشتغل على جهازك فقط (127.0.0.1)، ويفتح الواجهة في المتصفح.
الصورة مبتخرجش من جهازك. أول تشغيل بيحمّل نماذج EasyOCR (حوالي 100 ميجا) وبعدها بيشتغل بدون إنترنت.

Run:   python server.py            (add --debug to save raw OCR output to last_ocr_debug.json)
"""
import io
import json
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

APP_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(APP_DIR, "static")
RUN_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
MAX_BYTES = 30 * 1024 * 1024
MRZ_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
DEBUG = "--debug" in sys.argv or os.environ.get("DOCX_DEBUG") == "1"
ARABIC = range(0x0600, 0x0700)


# ---------------------------------------------------------------- image prep
def load_image(data, rot):
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im).convert("RGB")
    if rot % 360:
        im = im.rotate(-rot, expand=True)  # client rot is clockwise, PIL is counter-clockwise
    w, h = im.size
    scale = max(0.3, min(3.0, 2400 / max(w, h)))
    return im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def to_gray(im):
    return np.array(ImageOps.autocontrast(ImageOps.grayscale(im), cutoff=1))


def binarize(gray):
    if cv2 is None:
        return gray
    k = max(15, (gray.shape[1] // 40) | 1)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, k, 12)


# ---------------------------------------------------------------- OCR helpers
def is_arabic(s):
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum(ord(c) in ARABIC for c in letters) * 2 >= len(letters)


def split_words(box, text, conf):
    """Turn one detected line into word entries with proportionally split boxes.
    Arabic-dominant lines are laid out right-to-left (first logical word = rightmost)."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    toks = text.split()
    if not toks:
        return []
    weights = [len(t) + 0.6 for t in toks]
    total = sum(weights)
    width = x1 - x0
    out, pos = [], 0.0
    rtl = is_arabic(text)
    for t, w in zip(toks, weights):
        a = x0 + width * pos / total
        b = x0 + width * (pos + w) / total
        pos += w
        if rtl:
            a, b = x1 - (b - x0), x1 - (a - x0)
        out.append({"text": t, "confidence": round(float(conf) * 100, 1),
                    "bbox": {"x0": round(a, 1), "x1": round(b, 1), "y0": round(y0, 1), "y1": round(y1, 1)}})
    return out


def group_lines(items):
    """items: [(x0, x1, y0, y1, text)] -> text lines top-to-bottom, left-to-right."""
    if not items:
        return []
    hs = sorted(i[3] - i[2] for i in items)
    tol = max(6, 0.55 * hs[len(hs) // 2])
    items = sorted(items, key=lambda i: (i[2] + i[3]) / 2)
    rows, cur, cy = [], [], None
    for it in items:
        c = (it[2] + it[3]) / 2
        if cur and abs(c - cy) <= tol:
            cur.append(it)
            cy = sum((j[2] + j[3]) / 2 for j in cur) / len(cur)
        else:
            if cur:
                rows.append(cur)
            cur, cy = [it], c
    if cur:
        rows.append(cur)
    return [" ".join(i[4] for i in sorted(r, key=lambda i: i[0])) for r in rows]


class Engine:
    def __init__(self):
        self.ready = False
        self.error = None
        self.ar = None
        self.en = None
        self.lock = threading.Lock()

    def load(self):
        try:
            import easyocr  # heavy import, done in background
            kw = {"gpu": False, "verbose": False}
            models = os.path.join(APP_DIR, "models")  # bundled by build_exe.bat -> fully offline
            if os.path.isdir(models):
                kw.update(model_storage_directory=models, download_enabled=False)
            self.ar = easyocr.Reader(["ar", "en"], **kw)
            self.en = easyocr.Reader(["en"], **kw)
            self.ready = True
            print("Engine ready.")
        except Exception as e:  # pragma: no cover
            self.error = str(e)
            traceback.print_exc()

    def full_pass(self, gray):
        res = self.ar.readtext(gray, detail=1, paragraph=False)
        words, items, wsum, csum = [], [], 0, 0.0
        for box, text, conf in res:
            words.extend(split_words(box, text, conf))
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            items.append((min(xs), max(xs), min(ys), max(ys), text))
            wsum += len(text)
            csum += float(conf) * len(text)
        return {"text": "\n".join(group_lines(items)), "words": words,
                "confidence": round(100 * csum / wsum, 1) if wsum else 0.0, "_raw": res}

    def mrz_pass(self, gray, frac, use_bin):
        part = gray[int(gray.shape[0] * (1 - frac)):, :]
        if use_bin:
            part = binarize(part)
        part = cv2.copyMakeBorder(part, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255) if cv2 is not None else part
        res = self.en.readtext(part, detail=1, paragraph=False, allowlist=MRZ_CHARS, width_ths=2.0)
        items = []
        for box, text, _ in res:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            items.append((min(xs), max(xs), min(ys), max(ys), text.replace(" ", "")))
        return "\n".join(group_lines(items))

    def process(self, data, rot):
        t0 = time.time()
        im = load_image(data, rot)
        gray = to_gray(im)
        with self.lock:
            full = self.full_pass(gray)
            mrz = [self.mrz_pass(gray, 0.40, True), self.mrz_pass(gray, 0.62, False)]
        out = {"text": full["text"], "words": full["words"], "confidence": full["confidence"],
               "mrz": mrz, "seconds": round(time.time() - t0, 1)}
        if DEBUG:
            dbg = dict(out)
            dbg["raw"] = [[[list(map(float, p)) for p in b], t, float(c)] for b, t, c in full["_raw"]]
            with open(os.path.join(RUN_DIR, "last_ocr_debug.json"), "w", encoding="utf-8") as f:
                json.dump(dbg, f, ensure_ascii=False, indent=1)
        return out


ENGINE = Engine()


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "DocExtractor/2"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            return self._send(200, {"ready": ENGINE.ready, "error": ENGINE.error})
        if path in ("/", "/index.html"):
            with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/ocr":
            return self._send(404, {"error": "not found"})
        if not ENGINE.ready:
            return self._send(503, {"error": ENGINE.error or "engine not ready"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > MAX_BYTES:
                return self._send(413, {"error": "bad size"})
            data = self.rfile.read(n)
            rot = int(parse_qs(u.query).get("rot", ["0"])[0]) % 360
            return self._send(200, ENGINE.process(data, rot))
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})


def free_port(preferred=8765):
    for p in [preferred] + list(range(preferred + 1, preferred + 30)):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port")


def main():
    port = free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=ENGINE.load, daemon=True).start()
    url = "http://127.0.0.1:%d/" % port
    print("=" * 60)
    print(" Document Data Extractor  -  %s" % url)
    print(" Keep this window open while using the tool. Close it to stop.")
    print(" First run downloads OCR models (~100 MB), then works offline.")
    print("=" * 60)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
