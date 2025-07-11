import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Cargar modelo
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Cargar datos de prueba
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Predecir
y_pred = model.predict(X_test)

# Mostrar matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
    'Start', 'Stop', 'Counter clockwise spin', 'Clockwise spin', 'Increase speed', 'Decrease speed'
])
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confussion matrix - Gestures classifier")
plt.tight_layout()
plt.savefig('confussion_matrix.pdf', format='pdf')
print("Matrix saved as 'matriz_confusion.png'")