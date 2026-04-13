import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# STEP 1: DATA
X = np.random.rand(100, 64)
y = np.random.randint(0, 2, 100)

print("Dataset Shape:", X.shape)

# STEP 2: SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# STEP 3: MODEL
model = MLPClassifier(max_iter=500)
model.fit(X_train, y_train)

# STEP 4: PREDICT
y_pred = model.predict(X_test)

# STEP 5: ACCURACY
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# STEP 6: NEW PREDICTION
test = np.random.rand(1,64)
pred = model.predict(test)

print("\nPrediction:", "Disease Detected" if pred[0]==1 else "Normal")

# STEP 7: GRAPH
plt.plot(y_test, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.show()

input("Press Enter to exit...")