import os
import numpy as np
import cv2
import joblib
from datetime import datetime

class TrainingDataHandler:
    def __init__(self):
        self.training_data_dir = 'training_data'
        self.samples_dir = os.path.join(self.training_data_dir, 'samples')
        self.model_file = 'digit_model.joblib'
        self.initialize_directories()
    
    def initialize_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.training_data_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
    
    def save_training_sample(self, image_data, label):
        """Save a training sample with its label"""
        # Create a unique filename using timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'digit_{label}_{timestamp}.npy'
        filepath = os.path.join(self.samples_dir, filename)
        
        # Save the image data
        np.save(filepath, image_data)
        
        return filepath
    
    def load_training_data(self):
        """Load all training samples and their labels"""
        X_train = []
        y_train = []
        
        # Iterate through all saved samples
        for filename in os.listdir(self.samples_dir):
            if filename.endswith('.npy'):
                try:
                    # Extract label from filename (digit_label_timestamp.npy)
                    label = int(filename.split('_')[1])
                    
                    # Load image data with allow_pickle=True
                    filepath = os.path.join(self.samples_dir, filename)
                    image_data = np.load(filepath, allow_pickle=True)
                    
                    # Validate image data
                    if not isinstance(image_data, np.ndarray) or image_data.shape != (1, 64):
                        print(f"Skipping invalid data file: {filename}")
                        continue
                    
                    # Create one-hot encoded target
                    target = np.zeros(10)
                    target[label] = 1
                    
                    X_train.append(image_data)
                    y_train.append(target)
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
                    continue
        
        if X_train and y_train:
            return np.vstack(X_train), np.vstack(y_train)
        return np.array([], dtype=np.float32).reshape(0, 64), np.array([], dtype=np.float32).reshape(0, 10)
    
    def save_model(self, model_data):
        """Save the model with its training data"""
        joblib.dump(model_data, self.model_file)
    
    def load_model(self):
        """Load the model if it exists"""
        if os.path.exists(self.model_file):
            return joblib.load(self.model_file)
        return None