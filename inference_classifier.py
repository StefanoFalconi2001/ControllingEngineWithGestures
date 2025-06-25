import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load model and scaler
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

with open('./scaler.p', 'rb') as f:
    scaler = pickle.load(f)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

labels_dict = {0: 'Start', 1: 'Stop', 2: 'Counter clockwise spin',
               3: 'Clockwise spin', 4: 'Increase speed', 5: 'Decrease speed'}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                     mp_drawing_styles.get_default_hand_landmarks_style(),
                                     mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            x_ = np.array([lm.x for lm in hand_landmarks.landmark])
            y_ = np.array([lm.y for lm in hand_landmarks.landmark])

            data_aux = []
            for i in range(len(x_)):
                data_aux.append(x_[i])
                data_aux.append(y_[i])

            # Convert to numpy array and reshape for scaler
            data_aux = np.array(data_aux).reshape(1, -1)

            # Apply the same scaler used in training
            data_aux_scaled = scaler.transform(data_aux)

            prediction = model.predict(data_aux_scaled)
            probabilities = model.predict_proba(data_aux_scaled)

            predicted_character = labels_dict[int(prediction[0])]
            print(f"Prediction: {predicted_character} - Probabilities: {probabilities}")

            x1 = int(x_.min() * W) - 10
            y1 = int(y_.min() * H) - 10
            x2 = int(x_.max() * W) + 10
            y2 = int(y_.max() * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        print("ESC pressed. Exiting...")
        break

    if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
