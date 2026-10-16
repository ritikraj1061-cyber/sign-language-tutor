import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import threading
import pythoncom   # needed so Windows COM/SAPI5 speech works inside a background thread
from collections import deque

# -----------------------------
# TEXT TO SPEECH
# -----------------------------
is_speaking = False   # while True, gesture typing is paused

def speak_text(text):
    """Speak `text` on a background thread so the camera loop never freezes.

    A fresh pyttsx3 engine is created per call (rather than reusing one
    global engine across threads) — pyttsx3's run loop does not reliably
    restart on a shared engine when called from a new thread each time,
    which otherwise causes it to silently do nothing after the first call.
    """
    global is_speaking
    text = text.strip()
    if not text:
        return
    if is_speaking:
        return   # already talking — ignore this trigger

    def _run():
        global is_speaking
        is_speaking = True
        pythoncom.CoInitialize()   # required once per thread on Windows for SAPI5 to work
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[TTS ERROR] {e}")
        finally:
            pythoncom.CoUninitialize()
            is_speaking = False

    threading.Thread(target=_run, daemon=True).start()

# -----------------------------
# WORD SUGGESTIONS
# -----------------------------
# Small built-in common-word list (works fully offline).
# Feel free to replace/extend this with your own word list.
COMMON_WORDS = [
    "A","AM","ABOUT","ABOVE","AFTER","AGAIN","ALL","AN","AND","ANY","ARE","AS","AT",
    "BAD","BE","BEAUTIFUL","BECAUSE","BEEN","BEFORE","BEING","BELOW","BEST","BETWEEN",
    "BIG","BOY","BUT","BY","CAN","CANNOT","CAR","COME","COULD","DAY","DID","DO","DOES",
    "DOING","DOWN","EACH","EAT","FAMILY","FAR","FEW","FIND","FOR","FRIEND","FROM",
    "GIRL","GIVE","GO","GOOD","GOT","HAD","HAPPY","HAS","HAVE","HE","HELLO","HELP",
    "HER","HERE","HIM","HIS","HOME","HOUSE","HOW","I","IF","IN","INTO","IS","IT",
    "ITS","JUST","KNOW","LARGE","LIKE","LITTLE","LONG","LOVE","MAKE","MAN","MANY",
    "ME","MORE","MOST","MOTHER","MY","NAME","NEED","NEW","NO","NOT","NOW","OF","OFF",
    "OK","OLD","ON","ONE","ONLY","OR","OTHER","OUR","OUT","OVER","PLEASE","SAD",
    "SAME","SAY","SCHOOL","SEE","SHE","SMALL","SO","SOME","SORRY","STUDENT","SUCH",
    "TAKE","TEACHER","THAN","THANK","THANKS","THAT","THE","THEIR","THEM","THEN",
    "THERE","THESE","THEY","THIS","THOSE","TIME","TO","TODAY","TOO","UNDER","UP",
    "US","USE","VERY","WANT","WAS","WATER","WE","WELL","WENT","WERE","WHAT","WHEN",
    "WHERE","WHICH","WHO","WHY","WILL","WITH","WORK","WOULD","YES","YOU","YOUR",
]

def get_word_suggestions(prefix, limit=3):
    """Return up to `limit` common words that start with `prefix`."""
    prefix = prefix.upper()
    if not prefix:
        return []
    matches = [w for w in COMMON_WORDS if w.startswith(prefix) and w != prefix]
    return matches[:limit]

# -----------------------------
# LOAD MODELS
# -----------------------------
letter_model = pickle.load(open("model.pkl", "rb"))
symbol_model = pickle.load(open("model_symbols.pkl", "rb"))

# -----------------------------
# MEDIAPIPE
# -----------------------------
mp_hands = mp.solutions.hands
mp_selfie = mp.solutions.selfie_segmentation
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
segment = mp_selfie.SelfieSegmentation(model_selection=1)

cap = cv2.VideoCapture(0)

# -----------------------------
# LAYOUT
# -----------------------------
# +------------------+---------------------------+
# |                  |  A  B  C  D  E  F  G      |
# |   Camera Feed    |  H  I  J  K  L  M  N      |
# |    (left 55%)    |  O  P  Q  R  S  T  U      |
# |                  |  V  W  X  Y  Z  1  2      |
# +------------------+  3  4  5  6  7  8         |
# | Sentence:        |  9  0                     |
# | I AM A BOY_      +---------------------------+
# +------------------+

COLS    = 6
ROWS    = 6          # 26 letters + 10 digits = 36 signs, 6 per row = 6 rows exactly
THUMB   = 96
PAD_X   = 16
PAD_Y   = 46
GAP     = 10
LABEL_H = 18

# Exact width the 6-col grid needs
GUIDE_W = PAD_X * 2 + COLS * THUMB + (COLS - 1) * GAP

WIN_W   = 1400
CAM_W   = WIN_W - GUIDE_W        # camera gets everything else
# Exact height needed for all 6 rows (letters flow straight into digits)
GUIDE_H_NEEDED = PAD_Y + ROWS * (THUMB + GAP + LABEL_H)

CAM_H   = max(480, GUIDE_H_NEEDED + 10)   # camera height fits the full grid exactly
SENT_H  = 140
WIN_H   = CAM_H + SENT_H

SYMBOLS = ["SPACE", "COMMA", "FULLSTOP", "BACKSPACE"]
SYMBOL_CHAR = {
    "SPACE":    " ",
    "COMMA":    ",",
    "FULLSTOP": ".",
}

# Colours
C_BG       = (18,  18,  24)
C_PANEL    = (26,  26,  36)
C_ACCENT   = (80,  200, 140)
C_ACCENT2  = (80,  160, 255)
C_TEXT     = (235, 235, 235)
C_DIM      = (90,  90,  110)
C_DIVIDER  = (50,  50,  65)
C_TILE_BG  = (38,  38,  52)
C_TILE_HL  = (50,  130, 90)    # highlighted tile (last signed letter)
C_TILE_SYM = (40,  60,  100)   # symbol tiles

# -----------------------------
# STATE
# -----------------------------
sentence       = ""
stable_sign    = ""
stable_frames  = 0
STABLE_FRAMES  = 10
hand_removed   = True
sign_confirmed = False
last_sign      = ""
guide_label    = ""
prediction_buffer = deque(maxlen=12)
suggestions    = []

# Signs that share an identical handshape with a digit — the model can't
# tell them apart, so we ask the user which one they meant.
# sign -> (letter_choice, digit_choice)
AMBIGUOUS_SIGNS = {
    "O": ("O", "0"),
    "V": ("V", "2"),
    "W": ("W", "6"),
    "Z": ("Z", "1"),
}
awaiting_choice_sign = None   # e.g. "O" or "V" while a prompt is on screen

# Pre-load all sign images  (resize to thumb size)
LETTERS  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS   = [str(d) for d in range(1, 10)] + ["0"]   # 1..9 then 0, natural keyboard order
SYM_KEYS = ["SPACE", "COMMA", "FULLSTOP", "BACKSPACE"]
ALL_SIGNS = LETTERS + DIGITS + SYM_KEYS

sign_thumbs = {}
for s in ALL_SIGNS:
    img = cv2.imread(f"sign_images/{s}.png")
    if img is not None:
        sign_thumbs[s] = cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_LANCZOS4)

cv2.namedWindow("Sign Language", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Sign Language", WIN_W, WIN_H)

# -----------------------------
# HELPERS
# -----------------------------
def draw_text_wrapped(img, text, x, y, max_w, font, scale, color, thickness, line_gap=8):
    words = text.split(" ")
    line  = ""
    cy    = y
    for word in words:
        test = (line + " " + word).strip()
        (tw, th), _ = cv2.getTextSize(test, font, scale, thickness)
        if tw > max_w and line:
            cv2.putText(img, line, (x, cy), font, scale, color, thickness, cv2.LINE_AA)
            cy  += th + line_gap
            line = word
        else:
            line = test
    if line:
        cv2.putText(img, line, (x, cy), font, scale, color, thickness, cv2.LINE_AA)

def draw_tile_grid(ui, signs, tile_x0, tile_y0, y_bound, last_confirmed):
    """Draw a 6-wide grid of sign tiles starting at (tile_x0, tile_y0),
    clipped to y_bound. Shared by both the letters block and digits block."""
    for i, sign in enumerate(signs):

        col = i % COLS
        row = i // COLS

        tx = tile_x0 + col * (THUMB + GAP)
        ty = tile_y0 + row * (THUMB + GAP + LABEL_H)

        is_last = (sign == last_confirmed)

        bg     = C_TILE_HL if is_last else C_TILE_BG
        border = (140, 255, 180) if is_last else C_DIVIDER

        cv2.rectangle(ui, (tx, ty), (tx + THUMB, ty + THUMB), bg, -1)
        cv2.rectangle(ui, (tx, ty), (tx + THUMB, ty + THUMB), border, 1)

        if sign in sign_thumbs:
            thumb = sign_thumbs[sign]
            # Clip to available space in case tile goes near panel edge
            available_h = min(THUMB, y_bound - ty)
            available_w = min(THUMB, WIN_W - tx)
            if available_h > 0 and available_w > 0:
                ui[ty:ty+available_h, tx:tx+available_w] = thumb[:available_h, :available_w]
        else:
            (lw, lh), _ = cv2.getTextSize(sign, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
            cv2.putText(ui, sign,
                        (tx + (THUMB - lw)//2, ty + (THUMB + lh)//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        C_ACCENT if is_last else C_TEXT, 1, cv2.LINE_AA)

        lc = C_ACCENT if is_last else C_DIM
        (lw, _), _ = cv2.getTextSize(sign, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(ui, sign,
                    (tx + (THUMB - lw)//2, ty + THUMB + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, lc, 1, cv2.LINE_AA)

def draw_guide_panel(ui, last_confirmed):
    """Draw A-Z followed immediately by 0-9 in one continuous 6-wide grid."""

    gx0 = CAM_W + 2

    # Panel background
    cv2.rectangle(ui, (gx0, 0), (WIN_W, WIN_H), C_PANEL, -1)

    # Title
    cv2.putText(ui, "SIGN  REFERENCE  (A - Z, 0 - 9)",
                (gx0 + PAD_X, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_DIM, 1, cv2.LINE_AA)
    cv2.rectangle(ui, (gx0, 36), (WIN_W, 38), C_DIVIDER, -1)

    draw_tile_grid(ui, LETTERS + DIGITS, gx0 + PAD_X, PAD_Y, WIN_H, last_confirmed)

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (CAM_W, CAM_H))
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Background segmentation
    seg_result = segment.process(rgb)
    body_mask  = (seg_result.segmentation_mask > 0.5).astype(np.uint8) * 255
    hand_mask  = np.zeros((h, w), dtype=np.uint8)

    results   = hands.process(rgb)
    sign      = ""
    num_hands = 0

    if results.multi_hand_landmarks:

        num_hands         = len(results.multi_hand_landmarks)
        all_hand_features = []

        for hand_lm in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            wrist_x = hand_lm.landmark[0].x
            wrist_y = hand_lm.landmark[0].y
            coords  = []
            points  = []

            for lm in hand_lm.landmark:
                px = int(lm.x * w)
                py = int(lm.y * h)
                points.append([px, py])
                coords.append(lm.x - wrist_x)
                coords.append(lm.y - wrist_y)

            pts  = np.array(points)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(hand_mask, hull, 255)

            max_val = max(abs(v) for v in coords)
            if max_val != 0:
                all_hand_features.extend([v / max_val for v in coords])

        pred = ""
        if num_hands == 1 and len(all_hand_features) == 42:
            pred = letter_model.predict(
                np.array(all_hand_features).reshape(1, -1)
            )[0]
        elif num_hands == 2 and len(all_hand_features) == 84:
            pred = symbol_model.predict(
                np.array(all_hand_features).reshape(1, -1)
            )[0]

        if pred:
            prediction_buffer.append(pred)

        if prediction_buffer:
            sign = max(set(prediction_buffer), key=prediction_buffer.count)

        if sign == stable_sign:
            stable_frames += 1
        else:
            stable_sign   = sign
            stable_frames = 1

        if (stable_frames >= STABLE_FRAMES and hand_removed
                and not sign_confirmed and not is_speaking
                and awaiting_choice_sign is None):
            sign_confirmed = True
            hand_removed   = False
            last_sign      = sign
            guide_label    = sign

            if sign == "BACKSPACE":
                sentence = sentence[:-1]
            elif sign in SYMBOL_CHAR:
                sentence += SYMBOL_CHAR[sign]
            elif sign in AMBIGUOUS_SIGNS:
                # Don't type it yet — ask the user which one they meant.
                awaiting_choice_sign = sign
            else:
                sentence += sign

    else:
        prediction_buffer.clear()
        stable_sign    = ""
        stable_frames  = 0
        hand_removed   = True
        sign_confirmed = False

    # Blend background
    combined = cv2.bitwise_or(body_mask, hand_mask)
    combined = cv2.dilate(combined, np.ones((10, 10), np.uint8), iterations=1)
    combined = cv2.GaussianBlur(combined, (21, 21), 0)
    alpha    = (combined / 255.0)[:, :, None]
    gray_bg  = np.full_like(frame, 180)
    output   = (frame * alpha + gray_bg * (1 - alpha)).astype(np.uint8)

    # Hold progress bar
    if stable_frames > 0 and not sign_confirmed:
        pct   = min(stable_frames / STABLE_FRAMES, 1.0)
        bar_w = int(pct * (CAM_W - 40))
        cv2.rectangle(output, (20, CAM_H - 18), (CAM_W - 20, CAM_H - 6), (40,40,40), -1)
        col = C_ACCENT if pct < 1.0 else (255, 255, 80)
        cv2.rectangle(output, (20, CAM_H - 18), (20 + bar_w, CAM_H - 6), col, -1)

    # Ambiguous-sign confirmation prompt — shown until the user picks
    if awaiting_choice_sign is not None:
        letter_opt, digit_opt = AMBIGUOUS_SIGNS[awaiting_choice_sign]
        msg = f"Which one? [1] = Letter {letter_opt}    [2] = Number {digit_opt}"
        (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        bx, by = 12, CAM_H - 60
        cv2.rectangle(output, (bx - 6, by - mh - 12), (bx + mw + 12, by + 8), (15, 15, 15), -1)
        cv2.rectangle(output, (bx - 6, by - mh - 12), (bx + mw + 12, by + 8), C_ACCENT2, 1)
        cv2.putText(output, msg, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 80), 2, cv2.LINE_AA)

    # Small "speaking..." indicator
    if is_speaking:
        cv2.putText(output, "SPEAKING...", (CAM_W - 190, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 80), 2, cv2.LINE_AA)

    # Detected sign badge
    if sign:
        badge = sign if sign not in SYMBOL_CHAR and sign != "BACKSPACE" else f"[{sign}]"
        (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(output, (12, 10), (20 + bw, 20 + bh + 4), (15,15,15), -1)
        cv2.putText(output, badge, (16, 18 + bh),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_ACCENT, 2, cv2.LINE_AA)

    # Update word suggestions based on the word currently being typed
    current_word = sentence.split(" ")[-1] if sentence else ""
    suggestions  = get_word_suggestions(current_word)

    # ── Build UI ─────────────────────────────────────────────
    ui = np.full((WIN_H, WIN_W, 3), C_BG, dtype=np.uint8)

    # Camera
    ui[0:CAM_H, 0:CAM_W] = output

    # Vertical divider
    cv2.rectangle(ui, (CAM_W, 0), (CAM_W + 2, WIN_H), C_DIVIDER, -1)

    # Guide panel — all signs permanently
    draw_guide_panel(ui, guide_label)

    # Sentence strip
    sy = CAM_H
    cv2.rectangle(ui, (0, sy), (CAM_W, WIN_H), (20, 20, 28), -1)
    cv2.rectangle(ui, (0, sy), (CAM_W, sy + 3), C_ACCENT, -1)

    cv2.putText(ui, "Sentence:", (20, sy + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_DIM, 1, cv2.LINE_AA)

    # Suggestion chips (top-right of sentence strip): "1:WORD  2:WORD  3:WORD"
    if suggestions:
        chip_x = CAM_W - 20
        chip_texts = [f"{i+1}:{w}" for i, w in enumerate(suggestions)]
        for txt in reversed(chip_texts):
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            chip_x -= tw
            cv2.rectangle(ui, (chip_x - 8, sy + 10), (chip_x + tw + 6, sy + 10 + th + 10),
                          C_TILE_SYM, -1)
            cv2.rectangle(ui, (chip_x - 8, sy + 10), (chip_x + tw + 6, sy + 10 + th + 10),
                          C_ACCENT2, 1)
            cv2.putText(ui, txt, (chip_x - 3, sy + 10 + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1, cv2.LINE_AA)
            chip_x -= 16

    draw_text_wrapped(
        ui,
        sentence + ("_" if not sign_confirmed else ""),
        20, sy + 68,
        CAM_W - 30,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2, C_TEXT, 2
    )

    cv2.putText(ui, "[SPACE] speak sentence   [R] clear   [D] delete word   [1-3] pick suggestion   [Q] quit",
                (20, WIN_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_DIM, 1, cv2.LINE_AA)

    cv2.imshow("Sign Language", ui)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Any keypress while speaking is ignored below except quit (already handled above)
    if not is_speaking:

        # Resolve an ambiguous-sign choice first — takes priority over everything else
        if awaiting_choice_sign is not None and key in (ord('1'), ord('2')):
            letter_opt, digit_opt = AMBIGUOUS_SIGNS[awaiting_choice_sign]
            sentence += letter_opt if key == ord('1') else digit_opt
            awaiting_choice_sign = None

        elif key == ord(' '):
            # Keyboard spacebar = speak the whole sentence as printed so far,
            # punctuation and all. This is separate from the gesture-based
            # SPACE symbol, which still types a space character as before.
            speak_text(sentence)

        elif key == ord('r'):
            sentence    = ""
            guide_label = ""
            last_sign   = ""
            awaiting_choice_sign = None

        elif key == ord('d'):
            # Delete the last full word (and any trailing space)
            sentence = sentence.rstrip(" ")
            sentence = sentence.rsplit(" ", 1)[0] if " " in sentence else ""

        elif key in (ord('1'), ord('2'), ord('3')):
            idx = key - ord('1')
            if idx < len(suggestions):
                chosen = suggestions[idx]
                # Replace the word currently being typed with the chosen suggestion
                parts = sentence.split(" ")
                parts[-1] = chosen
                sentence = " ".join(parts) + " "

cap.release()
cv2.destroyAllWindows()