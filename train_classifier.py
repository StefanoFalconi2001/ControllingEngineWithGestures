import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Standardize the data (important for certain algorithms)
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
np.save('X_test.npy', x_test)
np.save('y_test.npy', y_test)

# Use StratifiedKFold for cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform cross-validation and print average accuracy
cv_scores = cross_val_score(model, data, labels, cv=kf)
cross_val_accuracy = cv_scores.mean()

# Train the model
model.fit(x_train, y_train)

# Predict the test set results
y_predict = model.predict(x_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_predict)
conf_matrix = confusion_matrix(y_test, y_predict)
class_report = classification_report(y_test, y_predict)

# Save the metrics to a .txt file
with open("metrics.txt", "w") as f:
    f.write(f"Cross-validation accuracy: {cross_val_accuracy:.4f}\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write("Confusion Matrix:\n")
    np.savetxt(f, conf_matrix, fmt='%d')
    f.write("\nClassification Report:\n")
    f.write(class_report)

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
