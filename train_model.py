from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Scale the data
X = X / 16.0

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
test_acc = model.score(X_test, y_test)
print(f'\nTest accuracy: {test_acc}')

# Save the model
joblib.dump(model, 'digit_model.joblib')
print('\nModel saved as digit_model.joblib')