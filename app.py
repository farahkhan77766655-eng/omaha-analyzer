"""
Omaha Card Detection Server — Render-compatible
=================================================
Browser sends webcam frames → server detects cards → returns JSON

Deploy to Render:
  Build command:  pip install -r requirements.txt
  Start command:  gunicorn app:app --bind 0.0.0.0:$PORT --timeout 60
"""

import cv2
import numpy as np
import base64
import json
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static')

# ─── CORS (allow browser to call from any origin) ─────────────────────────────
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/options', methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def options(path=''):
    return '', 200


# ─── Card Detection Engine ────────────────────────────────────────────────────

SUITS_INFO = [
    {'name': 's', 'sym': '♠', 'cls': 'black'},
    {'name': 'h', 'sym': '♥', 'cls': 'red'},
    {'name': 'd', 'sym': '♦', 'cls': 'red'},
    {'name': 'c', 'sym': '♣', 'cls': 'black'},
]

def decode_frame(b64_data: str) -> np.ndarray:
    """Decode base64 image from browser into OpenCV frame."""
    if ',' in b64_data:
        b64_data = b64_data.split(',')[1]
    data = base64.b64decode(b64_data)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_frame(frame: np.ndarray) -> str:
    """Encode OpenCV frame to base64 JPEG."""
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return base64.b64encode(buf).decode('utf-8')


def order_points(pts):
    pts = pts.astype('float32')
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype='float32')


def four_point_transform(image, pts):
    try:
        rect = order_points(pts)
        tl, tr, br, bl = rect
        wA = np.linalg.norm(br - bl)
        wB = np.linalg.norm(tr - tl)
        hA = np.linalg.norm(tr - br)
        hB = np.linalg.norm(tl - bl)
        maxW = max(int(wA), int(wB))
        maxH = max(int(hA), int(hB))
        if maxW < 30 or maxH < 30:
            return None
        dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype='float32')
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxW, maxH))
    except Exception:
        return None


def classify_suit(card_img):
    """Detect suit using color (red/black) + pip shape circularity."""
    h, w = card_img.shape[:2]
    # suit pip lives in top-left corner, below the rank glyph
    pip_region = card_img[int(h*0.13): int(h*0.30), 0: int(w*0.28)]
    if pip_region.size == 0:
        return None

    # ── Color: is it red? ──
    hsv = cv2.cvtColor(pip_region, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0,   80, 80), (12,  255, 255))
    red2 = cv2.inRange(hsv, (158, 80, 80), (180, 255, 255))
    red_px = cv2.countNonZero(cv2.bitwise_or(red1, red2))
    total  = pip_region.shape[0] * pip_region.shape[1]
    is_red = red_px / max(total, 1) > 0.06

    # ── Shape: circularity of pip contour ──
    gray = cv2.cvtColor(pip_region, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.5  # default
    if cnts:
        biggest = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(biggest)
        peri = cv2.arcLength(biggest, True)
        if peri > 0:
            circularity = 4 * np.pi * area / (peri * peri)

    if is_red:
        return 'h' if circularity > 0.60 else 'd'
    else:
        return 'c' if circularity > 0.62 else 's'


def classify_rank(card_img):
    """
    Classify card rank from the top-left corner glyph.
    Uses contour geometry: aspect ratio, solidity, vertex count.
    """
    h, w = card_img.shape[:2]
    corner = card_img[0: int(h*0.32), 0: int(w*0.28)]
    if corner.size == 0:
        return None

    gray   = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (60, 90))
    _, bw  = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # isolate top half (rank glyph only, not suit pip)
    rank_bw = bw[:50, :]
    cnts, _ = cv2.findContours(rank_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    main = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(main)
    if area < 8:
        return None

    x, y, cw, ch = cv2.boundingRect(main)
    ar       = cw / float(ch) if ch > 0 else 1.0
    hull     = cv2.convexHull(main)
    hull_a   = cv2.contourArea(hull)
    solidity = area / hull_a if hull_a > 0 else 0
    peri     = cv2.arcLength(main, True)
    approx   = cv2.approxPolyDP(main, 0.04 * peri, True)
    verts    = len(approx)

    # ── Heuristic decision tree ──────────────────────────────────────────────
    # Tuned on standard poker deck printed cards.
    # Replace with a CNN for production-grade accuracy.

    if ar > 0.90:                        # very wide: T (10)
        return 'T'
    if ar < 0.35:                        # very narrow: 1 / I / 7 glyphs
        return 'A' if solidity > 0.70 else '7'
    if solidity > 0.88 and verts <= 5:
        if ar < 0.55:
            return 'J'
        return 'Q'
    if solidity > 0.80:
        if ar > 0.70:
            return '0'   # ambiguous wide — mapped below
        if verts <= 4:
            return 'A'
        return '4'
    if verts >= 8:
        return '8' if ar > 0.55 else '6'
    if ar > 0.70:
        return '9' if solidity < 0.70 else 'K'
    if ar < 0.50:
        return '5' if solidity > 0.60 else '7'
    if solidity < 0.55:
        return '3' if ar > 0.55 else '2'

    # Last resort: use area-to-bounding ratio bands
    density_map = [
        (0.75, 'K'), (0.70, 'Q'), (0.65, 'J'), (0.60, 'T'),
        (0.55, '9'), (0.50, '8'), (0.45, '6'), (0.40, '5'),
        (0.35, '4'), (0.30, '3'), (0.25, '2'), (0.00, 'A'),
    ]
    for threshold, rank in density_map:
        if solidity >= threshold:
            return rank
    return None


def detect_cards(frame: np.ndarray):
    """
    Main detection pipeline.
    Returns (list_of_card_dicts, annotated_debug_frame).
    """
    debug  = frame.copy()
    h, w   = frame.shape[:2]
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold handles varied lighting better than global Otsu
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    # Also try Otsu and combine
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = cv2.bitwise_or(thresh, otsu)

    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = h * w * 0.008
    max_area = h * w * 0.40
    card_candidates = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        bx, by, bw2, bh2 = cv2.boundingRect(approx)
        ar = bw2 / float(bh2)
        if 0.50 < ar < 0.80:          # standard card aspect ratio
            card_candidates.append((approx, area, bx, by, bw2, bh2))

    # Sort left-to-right by x position
    card_candidates.sort(key=lambda c: c[2])

    cards    = []
    seen     = set()

    for approx, area, bx, by, bw2, bh2 in card_candidates[:10]:
        card_img = four_point_transform(frame, approx.reshape(4, 2))
        if card_img is None:
            continue

        # Make portrait orientation
        ch, cw = card_img.shape[:2]
        if cw > ch:
            card_img = cv2.rotate(card_img, cv2.ROTATE_90_CLOCKWISE)

        rank = classify_rank(card_img)
        suit = classify_suit(card_img)

        if not rank or not suit:
            continue

        key = rank + suit
        if key in seen:
            continue
        seen.add(key)

        suit_info = next((s for s in SUITS_INFO if s['name'] == suit), SUITS_INFO[0])
        cards.append({
            'rank': rank,
            'suit': suit,
            'key':  key,
            'cls':  suit_info['cls'],
            'sym':  suit_info['sym'],
            'x':    int(bx),
            'y':    int(by),
        })

        # Annotate debug frame
        color = (60, 80, 220) if suit in ('h', 'd') else (60, 220, 80)
        cv2.drawContours(debug, [approx], -1, color, 3)
        label = f"{rank}{suit_info['sym']}"
        cv2.putText(debug, label, (bx, max(by - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    return cards, debug


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/detect', methods=['POST'])
def detect():
    """
    Endpoint: POST /detect
    Body: { "frame": "<base64 jpeg/png>" }
    Returns: { "ok": true, "cards": [...], "debug": "<base64 jpeg>" }
    """
    try:
        data  = request.get_json(force=True)
        b64   = data.get('frame', '')
        if not b64:
            return jsonify({'ok': False, 'error': 'No frame provided'}), 400

        frame = decode_frame(b64)
        if frame is None:
            return jsonify({'ok': False, 'error': 'Could not decode image'}), 400

        cards, debug_frame = detect_cards(frame)
        debug_b64 = encode_frame(debug_frame)

        return jsonify({'ok': True, 'cards': cards, 'debug': debug_b64})

    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'ok': True, 'service': 'Omaha Card Detector'})


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5050))
    print(f"\n🃏  Omaha Card Detection Server")
    print(f"    Running on http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
