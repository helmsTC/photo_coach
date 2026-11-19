"""Local web app.

    python -m photocoach.app            # then open http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import os
import time
import uuid
from collections import OrderedDict

from flask import Flask, Response, abort, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 120 * 1024 * 1024

_engine = None
_cache: "OrderedDict[str, dict]" = OrderedDict()


def engine():
    global _engine
    if _engine is None:
        from .engine import Engine
        _engine = Engine()
    return _engine


@app.get("/")
def index():
    return send_from_directory(os.path.join(HERE, "static"), "index.html")


@app.post("/api/compare")
def compare():
    fa, fb = request.files.get("a"), request.files.get("b")
    if not fa or not fb:
        return jsonify(error="Upload two photos."), 400
    na = (request.form.get("name_a") or "Photo A").strip()[:40] or "Photo A"
    nb = (request.form.get("name_b") or "Photo B").strip()[:40] or "Photo B"
    if na == nb:
        na, nb = na + " (1)", nb + " (2)"
    t0 = time.time()
    try:
        result, images = engine().compare(fa.read(), fb.read(), na, nb)
    except Exception as e:  # unreadable image etc.
        app.logger.exception("compare failed")
        return jsonify(error=f"Could not analyse these files: {e}"), 400
    return _finish(result, images, t0)


@app.get("/api/sample")
def sample():
    """Compare two files from data/demo (for trying the app without uploading)."""
    demo = os.path.join(os.path.dirname(HERE), "data", "demo")
    names = [os.path.basename(request.args.get(k, "")) for k in ("a", "b")]
    paths = [os.path.join(demo, n) for n in names]
    if not all(n and os.path.isfile(p) for n, p in zip(names, paths)):
        return jsonify(error="Sample files not found in data/demo."), 404
    t0 = time.time()
    result, images = engine().compare(open(paths[0], "rb").read(), open(paths[1], "rb").read(),
                                      request.args.get("name_a", "My photo"), request.args.get("name_b", "Reference"))
    return _finish(result, images, t0)


def _finish(result, images, t0):
    rid = uuid.uuid4().hex[:12]
    _cache[rid] = images
    while len(_cache) > 6:
        _cache.popitem(last=False)
    result["id"] = rid
    result["seconds"] = round(time.time() - t0, 1)
    return jsonify(result)


@app.get("/img/<rid>/<name>.jpg")
def img(rid, name):
    data = _cache.get(rid, {}).get(name)
    if data is None:
        abort(404)
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    print("Loading models (first start downloads ~2.3 GB of backbone weights)...", flush=True)
    engine()
    print(f"Ready: http://{args.host}:{args.port}", flush=True)
    app.run(args.host, args.port, threaded=True)


if __name__ == "__main__":
    main()
