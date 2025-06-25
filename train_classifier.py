import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# === Load the data ===
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels']).astype(int)  # ✅ Ensure integer labels

# === Check class distribution ===
label_counts = Counter(labels)
print("✅ Label distribution:", label_counts)

# === Check if all classes have at least 2 samples (needed for stratify) ===
min_class_count = min(label_counts.values())
use_stratify = min_class_count >= 2

if not use_stratify:
    print("⚠️ Some class has fewer than 2 samples. Stratified splitting is disabled.")
else:
    print("✅ All classes have enough samples for stratified split.")

# === Standardize the data ===
scaler = StandardScaler()
data = scaler.fit_transform(data)

# === PCA Visualization ===
pca = PCA(n_components=2)
projected = pca.fit_transform(data)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(projected[:, 0], projected[:, 1], c=labels, cmap='tab10', s=10)
plt.title("PCA of the Dataset")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(scatter, label="Class")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_visualization.png")  # Save the plot
plt.close()

# === Train-test split ===
if use_stratify:
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )
else:
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True
    )

# Optionally save test set
np.save('X_test.npy', x_test)
np.save('y_test.npy', y_test)

# === Cross-validation ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

cv_scores = cross_val_score(model, data, labels, cv=kf)
cross_val_accuracy = cv_scores.mean()

# === Train the model ===
model.fit(x_train, y_train)

# === Evaluate on test set ===
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# === Save metrics to file ===
with open("metrics.txt", "w") as f:
    f.write(f"Cross-validation accuracy: {cross_val_accuracy:.4f}\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write("Confusion Matrix:\n")
    np.savetxt(f, conf_matrix, fmt='%d')
    f.write("\nClassification Report:\n")
    f.write(class_report)

# === Save the trained model ===
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

with open('scaler.p', 'wb') as f:
    pickle.dump(scaler, f)


print("✅ Training complete. Model and metrics saved.")
