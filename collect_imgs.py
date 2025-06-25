import os
import cv2
import random

DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

num_classes = 6
images_per_class = 1000

# Function to apply augmentation (rotation + brightness)
def augment_image(img):
    # Random rotation
    angle = random.randint(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))

    # Random brightness
    value = random.randint(-40, 40)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.add(hsv[..., 2], value)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img

cap = cv2.VideoCapture(0)

for class_id in range(num_classes):
    class_dir = os.path.join(DATA_DIR, str(class_id))
    os.makedirs(class_dir, exist_ok=True)

    print(f"\n🔴 Ready to capture CLASS {class_id}.")
    print("📸 Press 'c' to capture an image.")
    print("⏭ Press 'e' to skip this class.")

    counter = 0

    while counter < images_per_class:
        ret, frame = cap.read()
        display_frame = frame.copy()

        cv2.putText(display_frame, f'Class {class_id} - Image {counter+1}/{images_per_class}',
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Manual Capture', display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            augmented = augment_image(frame)
            filename = os.path.join(class_dir, f'{counter}.jpg')
            cv2.imwrite(filename, augmented)
            print(f'✅ Saved image {counter+1} for class {class_id}')
            counter += 1

        elif key == ord('e'):
            print(f'⏭ Skipped class {class_id} by user.')
            break

print("🎉 Image capture completed.")
cap.release()
cv2.destroyAllWindows()
