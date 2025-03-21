import kagglehub
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib

# Download MNIST dataset
print('Downloading MNIST dataset...')
path = kagglehub.dataset_download('hojjatk/mnist-dataset')
print('Dataset downloaded to:', path)

# Load and preprocess MNIST data
print('Loading and preprocessing data...')
train_images = np.load(f'{path}/train_images.npy')
train_labels = np.load(f'{path}/train_labels.npy')
test_images = np.load(f'{path}/test_images.npy')
test_labels = np.load(f'{path}/test_labels.npy')

# Reshape and scale the data
X_train = train_images.reshape(train_images.shape[0], -1) / 255.0
X_test = test_images.reshape(test_images.shape[0], -1) / 255.0
y_train = train_labels
y_test = test_labels

# Create and train an improved model
print('Training model...')
model = MLPClassifier(
    hidden_layer_sizes=(256, 128),  # Larger network for more complex patterns
    max_iter=50,                    # More iterations for better convergence
    learning_rate_init=0.001,       # Adjusted learning rate
    batch_size=128,                 # Efficient batch size
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate the model
test_acc = model.score(X_test, y_test)
print(f'\nTest accuracy: {test_acc}')

# Save the model
joblib.dump(model, 'digit_model.joblib')
print('\nModel saved as digit_model.joblib')