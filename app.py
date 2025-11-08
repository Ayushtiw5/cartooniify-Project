import sys
import os, uuid, numpy as np
try:
    import cv2
except ImportError:
    print('\nMissing required dependency: opencv-python (provides `cv2`).')
    print('Install it with:')
    print('    python -m pip install opencv-python')
    # Exit early so the user sees the message instead of a traceback
    sys.exit(1)
from flask import Flask, render_template, request, url_for, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
import sys
import os
import uuid
import json
import time
from datetime import datetime
import numpy as np

try:
    import cv2
except ImportError:
    print('\nMissing required dependency: opencv-python (provides `cv2`).')
    print('Install it with:')
    print('    python -m pip install opencv-python')
    sys.exit(1)

from flask import Flask, render_template, request, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_DIR = os.path.join(app.root_path, "static", "uploads")
RESULT_DIR = os.path.join(app.root_path, "static", "results")
HISTORY_FILE = os.path.join(RESULT_DIR, "history.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def _append_history(entry, limit=60):
    try:
        hist = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                hist = json.load(f)
        hist.insert(0, entry)
        hist = hist[:limit]
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    except Exception:
        # best-effort only
        pass


def animeify(bgr, sat_mult=1.15, edge_dilate=1, sigmaColor=75, sigmaSpace=75):
    """Produce an anime-style/cartoonified version of a BGR image.

    Parameters:
    - bgr: input BGR image (numpy array)
    - sat_mult: saturation multiplier (float)
    - edge_dilate: how many times to dilate the edge mask (int)

    Returns a BGR image (same dtype as input) or None if input invalid.
    """
    if bgr is None:
        return None

    img = bgr.copy()

    # Smooth while preserving edges
    # bilateral filter with tunable parameters
    for _ in range(2):
        img = cv2.bilateralFilter(img, d=9, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    # Color simplification (mean-shift)
    try:
        img_color = cv2.pyrMeanShiftFiltering(img, sp=20, sr=40)
    except Exception:
        img_color = img

    # Saturation boost
    try:
        sm = float(sat_mult)
    except Exception:
        sm = 1.15
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * sm, 0, 255)
    img_color = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Edge detection on the original to preserve strong lines
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 5)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Dilate edges to make them more ink-like
    try:
        dil = int(edge_dilate)
    except Exception:
        dil = 1
    if dil > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=dil)

    # Combine edges with simplified color image
    cartoon = cv2.bitwise_and(img_color, edges)

    # Mild sharpening
    blur = cv2.GaussianBlur(cartoon, (0, 0), sigmaX=1.0)
    cartoon = cv2.addWeighted(cartoon, 1.2, blur, -0.2, 0)

    return cartoon


def cartoonify(bgr):
    """Simple cartoonify used as alternate mode."""
    if bgr is None:
        return None
    img = cv2.bilateralFilter(bgr, 9, 250, 250)
    img = cv2.bilateralFilter(img, 9, 250, 250)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 5)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(img, edges)


def sketchify(bgr, ksize=21):
    """Produce a pencil-sketch like result (grayscale) using color dodge technique."""
    if bgr is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    if ksize % 2 == 0:
        ksize += 1
    blur = cv2.GaussianBlur(inv, (ksize, ksize), 0)
    # color dodge: result = gray / (255 - blur)
    denom = 255 - blur
    denom[denom == 0] = 1
    sketch = cv2.divide(gray, denom, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def xdogify(bgr, sigma=1.0, k=1.6, p=10.0, epsilon=0.1):
    """Approximate xDoG: DoG + soft threshold to produce ink-like edges."""
    if bgr is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma * k)
    dog = g1 - p * g2
    # normalize
    dog = (dog - dog.min()) / (dog.max() - dog.min() + 1e-9)
    # soft threshold
    edges = np.where(dog < epsilon, 1.0, 1.0 - np.tanh((dog - epsilon) * 10.0))
    edges = (edges * 255).astype(np.uint8)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # combine with desaturated color for stylized look
    color = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    color[..., 1] = (color[..., 1] * 0.6).astype(color.dtype)
    color_bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
    return cv2.bitwise_and(color_bgr, edges)


@app.route('/process', methods=['POST'])
def process_api():
    """AJAX-friendly processing endpoint. Returns JSON with URLs."""
    if 'image' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400
    f = request.files['image']
    if not f or not f.filename:
        return jsonify({'error': 'invalid file'}), 400

    mode = request.form.get('mode', 'anime')
    sat = request.form.get('sat', None)
    dil = request.form.get('dilate', None)
    # new params
    sigmaColor = request.form.get('sigmaColor', None)
    sigmaSpace = request.form.get('sigmaSpace', None)
    blockSize = request.form.get('blockSize', None)
    C = request.form.get('C', None)
    downscale = request.form.get('scale', None)
    ksize = request.form.get('ksize', None)
    x_sigma = request.form.get('x_sigma', None)
    x_k = request.form.get('x_k', None)

    safe = secure_filename(f.filename)
    uid = uuid.uuid4().hex
    in_name = f"{uid}{os.path.splitext(safe)[1].lower()}"
    in_path = os.path.join(UPLOAD_DIR, in_name)
    f.save(in_path)

    bgr = cv2.imread(in_path)
    # optionally downscale for performance
    scale = 1.0
    try:
        if downscale:
            scale = float(downscale)
            if scale <= 0 or scale > 1.0:
                scale = 1.0
    except Exception:
        scale = 1.0
    if scale != 1.0:
        h, w = bgr.shape[:2]
        bgr_small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        bgr_small = bgr

    if mode == 'cartoon':
        out = cartoonify(bgr_small)
    elif mode == 'sketch':
        try:
            k = int(ksize) if ksize else 21
        except Exception:
            k = 21
        out = sketchify(bgr_small, ksize=k)
    elif mode == 'xdog':
        try:
            xs = float(x_sigma) if x_sigma else 1.0
            xk = float(x_k) if x_k else 1.6
        except Exception:
            xs, xk = 1.0, 1.6
        out = xdogify(bgr_small, sigma=xs, k=xk)
    else:
        try:
            sc = float(sigmaColor) if sigmaColor else 75
            ss = float(sigmaSpace) if sigmaSpace else 75
        except Exception:
            sc, ss = 75, 75
        try:
            ed = int(dil) if dil else 1
        except Exception:
            ed = 1
        try:
            sat_f = float(sat) if sat else 1.15
        except Exception:
            sat_f = 1.15
        out = animeify(bgr_small, sat_mult=sat_f, edge_dilate=ed, sigmaColor=sc, sigmaSpace=ss)

    # If processed at smaller scale, upscale back to original size for consistent download
    if scale != 1.0:
        out = cv2.resize(out, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    out_name = f"{uid}_result.jpg"
    out_path = os.path.join(RESULT_DIR, out_name)
    cv2.imwrite(out_path, out)

    # append to history (timestamp and filenames)
    entry = {
        'id': uid,
        'input': f'uploads/{in_name}',
        'output': f'results/{out_name}',
        'time': datetime.utcnow().isoformat() + 'Z'
    }
    _append_history(entry)

    return jsonify({
        'input_url': url_for('static', filename=f'uploads/{in_name}'),
        'output_url': url_for('static', filename=f'results/{out_name}'),
        'id': uid
    })


@app.route('/', methods=['GET', 'POST'])
def index():
    # keep simple: template handles the interactive frontend; server-side POST is optional fallback
    if request.method == 'POST' and 'image' in request.files:
        # fallback synchronous processing
        f = request.files['image']
        if f and f.filename:
            safe = secure_filename(f.filename)
            uid = uuid.uuid4().hex
            in_name = f"{uid}{os.path.splitext(safe)[1].lower()}"
            in_path = os.path.join(UPLOAD_DIR, in_name)
            f.save(in_path)
            bgr = cv2.imread(in_path)
            out = animeify(bgr)
            out_name = f"{uid}_result.jpg"
            out_path = os.path.join(RESULT_DIR, out_name)
            cv2.imwrite(out_path, out)
            return render_template('index.html',
                                   input_url=url_for('static', filename=f'uploads/{in_name}'),
                                   output_url=url_for('static', filename=f'results/{out_name}'))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)


@app.route('/download/<path:filename>')
def download_result(filename):
    # serve result files as attachment for download
    # security: ensure filename doesn't contain path traversal
    return send_from_directory(RESULT_DIR, filename, as_attachment=True)


@app.route('/gallery')
def gallery_api():
    # return history JSON for client-side gallery
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                hist = json.load(f)
        else:
            hist = []
    except Exception:
        hist = []
    return jsonify(hist)