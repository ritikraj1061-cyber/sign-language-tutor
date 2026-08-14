import cv2
import mediapipe as mp
import csv

# -----------------------------
# MediaPipe
# -----------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# Camera
# -----------------------------

cap = cv2.VideoCapture(0)

label = input(
    "Enter symbol label (SPACE, COMMA, FULLSTOP, BACKSPACE): "
)

sample_count = 0
max_samples = 300

with open("dataset_symbols.csv", "a", newline="") as f:

    writer = csv.writer(f)

    while sample_count < max_samples:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        # Need EXACTLY 2 hands
        if (
            results.multi_hand_landmarks
            and len(results.multi_hand_landmarks) == 2
        ):

            all_features = []

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y

                coords = []

                for lm in hand_landmarks.landmark:

                    x = lm.x - wrist_x
                    y = lm.y - wrist_y

                    coords.append(x)
                    coords.append(y)

                max_value = max(
                    abs(v) for v in coords
                )

                normalized = []

                for v in coords:
                    normalized.append(v / max_value)

                all_features.extend(normalized)

            # 42 + 42 = 84 features
            if len(all_features) == 84:

                row = all_features + [label]

                writer.writerow(row)

                sample_count += 1

        # -----------------------------
        # UI
        # -----------------------------

        cv2.putText(
            frame,
            f"{label}: {sample_count}/{max_samples}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            "SHOW BOTH HANDS",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

        cv2.imshow(
            "Collect Symbols",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()