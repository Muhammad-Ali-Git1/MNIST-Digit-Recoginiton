from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
import numpy as np
import cv2

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data
y = mnist.target.astype(int)

def preprocess(img_flat):
    img = img_flat.reshape(28, 28).astype(np.float32)
    img = img / 255.0
    if img.mean() > 0.5:
        img = 1 - img
    img = (img > 0.2).astype(np.float32)

    coords = cv2.findNonZero((img * 255).astype(np.uint8))
    if coords is not None:
        x, y_coord, w, h = cv2.boundingRect(coords)
        digit = img[y_coord:y_coord+h, x:x+w]
        digit = np.pad(digit, 4, mode='constant', constant_values=0)
        img   = cv2.resize(digit, (28, 28))

    return img.flatten()
print("Preprocessing images... (takes ~1 minute)")
X = np.array([preprocess(x) for x in X])
print("Done!")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ml = MLPClassifier(hidden_layer_sizes=(128, 64),activation='relu',max_iter=50,verbose=True)
ml.fit(X_train, y_train)
joblib.dump(ml, "digit_model.pkl")
print("Model saved successfully!")
