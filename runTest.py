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
    'Marcha', 'Paro', 'Giro Antihorario', 'Giro Horario', 'Aumentar Velocidad', 'Disminuir Velocidad'
])
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Matriz de Confusión - Clasificador de Gestos")
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=300)
print("Matriz de confusión guardada como 'matriz_confusion.png'")