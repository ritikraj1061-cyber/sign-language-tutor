import cv2
import mediapipe as mp
import csv

# Load MediaPipe hand detection model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# For drawing hand skeleton
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Ask which letter we are recording
label = input("Enter letter label (A-Z): ")

sample_count = 0
max_samples = 500

with open("dataset.csv", "a", newline="") as f:

    writer = csv.writer(f)

    while sample_count < max_samples:

        ret, frame = cap.read()
        frame = cv2.flip(frame,1)

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                coords = []

                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y

                for lm in hand_landmarks.landmark:

                    x = lm.x - wrist_x
                    y = lm.y - wrist_y

                    coords.append(x)
                    coords.append(y)

                max_value = max(abs(val) for val in coords)

                features = []

                for val in coords:
                    features.append(val / max_value)

                features.append(label)

                writer.writerow(features)

                sample_count += 1

        cv2.putText(
            frame,
            f"Samples: {sample_count}/500",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()