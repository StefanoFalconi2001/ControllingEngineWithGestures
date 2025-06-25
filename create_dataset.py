import os
import pickle
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    print(f"Processing class '{dir_}'")

    for img_path in os.listdir(dir_path):
        img_file = os.path.join(dir_path, img_path)
        img = cv2.imread(img_file)
        if img is None:
            print(f"Warning: could not read image {img_file}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                data_aux = []
                for i in range(len(x_)):
                    # Normalize coordinates to [0,1]
                    data_aux.append((x_[i] - min(x_)) / (max(x_) - min(x_) + 1e-6))
                    data_aux.append((y_[i] - min(y_)) / (max(y_) - min(y_) + 1e-6))

                if len(data_aux) == 42:  # 21 landmarks * 2 coords
                    data.append(data_aux)
                    labels.append(int(dir_))  # Save label as int
                else:
                    print(f"Warning: Unexpected data length {len(data_aux)} in {img_file}")

        else:
            print(f"No hand landmarks found in {img_file}")

print(f"Total samples collected: {len(data)}")

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset saved to 'data.pickle'")
