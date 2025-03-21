import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image, ImageDraw
import joblib
import io
import os
import pyttsx3
from training_data_handler import TrainingDataHandler

class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition")
        self.root.configure(bg='#f0f0f0')
        self.root.resizable(False, False)
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Initialize instance variables
        self.current_processed_image = None
        
        # Initialize training data handler
        self.training_handler = TrainingDataHandler()
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure('TButton', padding=6, relief='flat', background='#2196F3')
        self.style.configure('TLabel', font=('Helvetica', 12), background='#f0f0f0')
        self.style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'), background='#f0f0f0')
        self.style.configure('Result.TLabel', font=('Helvetica', 16, 'bold'), foreground='#2196F3', background='#f0f0f0')
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10 10 10 10")
        self.main_frame.grid(row=0, column=0, sticky="".join((tk.W, tk.E, tk.N, tk.S)))
        
        # Header
        self.header_label = ttk.Label(self.main_frame, text='Draw a digit (0-9)', style='Header.TLabel')
        self.header_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Initialize drawing variables
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # Create canvas with border and shadow effect
        self.canvas_frame = ttk.Frame(self.main_frame, padding=2)
        self.canvas_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=280,
            height=280,
            bg='black',
            highlightthickness=2,
            highlightbackground='#2196F3'
        )
        self.canvas.grid(row=0, column=0)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # Create button frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Create buttons with icons (using Unicode characters)
        self.clear_btn = ttk.Button(self.button_frame, text='🗑️ Clear', command=self.clear_canvas)
        self.clear_btn.grid(row=0, column=0, padx=10)
        
        self.predict_btn = ttk.Button(self.button_frame, text='🔍 Predict', command=self.predict_digit)
        self.predict_btn.grid(row=0, column=1, padx=10)
        
        # Create result label with custom style
        self.result_label = ttk.Label(
            self.main_frame,
            text='Draw a digit above',
            style='Result.TLabel',
            anchor='center'
        )
        self.result_label.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Create training frame with improved layout
        self.training_frame = ttk.Frame(self.main_frame)
        self.training_frame.grid(row=4, column=0, columnspan=2, pady=5)
        self.training_frame.grid_remove()  # Hide initially
        
        # Create training components with better styling
        ttk.Label(
            self.training_frame,
            text='Is this correct? If not, enter the right digit:',
            style='TLabel'
        ).pack(side=tk.TOP, pady=(0, 5))
        
        self.correct_digit = ttk.Spinbox(
            self.training_frame,
            from_=0,
            to=9,
            width=5,
            font=('Helvetica', 12)
        )
        self.correct_digit.pack(side=tk.TOP, pady=5)
        
        self.train_btn = ttk.Button(
            self.training_frame,
            text='✓ Train Model',
            command=self.manual_train
        )
        self.train_btn.pack(side=tk.TOP, pady=5)
        
        # Add tooltips
        self.create_tooltip(self.canvas, 'Draw a digit using your mouse')
        self.create_tooltip(self.clear_btn, 'Clear the canvas')
        self.create_tooltip(self.predict_btn, 'Predict the drawn digit')
        self.create_tooltip(self.correct_digit, 'Enter the correct digit (0-9)')
        
        # Initialize neural network model
        self.initialize_model()
    
    def create_tooltip(self, widget, text):
        widget.tooltip_text = text
        widget.bind('<Enter>', self.show_tooltip)
        widget.bind('<Leave>', self.hide_tooltip)
    
    def show_tooltip(self, event):
        widget = event.widget
        tooltip = tk.Label(
            self.root,
            text=widget.tooltip_text,
            background='#2196F3',
            foreground='white',
            relief='solid',
            borderwidth=1,
            font=('Helvetica', 10)
        )
        tooltip.place_forget()
        
        x = widget.winfo_rootx() + widget.winfo_width()//2
        y = widget.winfo_rooty() + widget.winfo_height() + 5
        
        tooltip.place(x=x, y=y, anchor='n')
        widget.tooltip = tooltip
    
    def hide_tooltip(self, event):
        widget = event.widget
        if hasattr(widget, 'tooltip'):
            widget.tooltip.destroy()
            del widget.tooltip
        
    def initialize_model(self):
        # Simple neural network with one hidden layer
        self.weights = {}
        self.biases = {}
        
        # Try to load existing model and dataset
        model_data = self.training_handler.load_model()
        if model_data and isinstance(model_data, dict):
            self.weights = model_data.get('weights', {})
            self.biases = model_data.get('biases', {})
            # Load training data from saved samples
            self.X_train, self.y_train = self.training_handler.load_training_data()
            if not isinstance(self.X_train, np.ndarray) or self.X_train.size == 0 or not isinstance(self.y_train, np.ndarray) or self.y_train.size == 0:
                self._initialize_new_model()
        else:
            self._initialize_new_model()
    
    def _initialize_new_model(self):
        # Initialize weights and biases with small random values
        # Input layer (64) to hidden layer (32)
        self.weights['h1'] = np.random.randn(64, 32) * 0.01
        self.biases['h1'] = np.zeros((1, 32))
        
        # Hidden layer (32) to output layer (10)
        self.weights['out'] = np.random.randn(32, 10) * 0.01
        self.biases['out'] = np.zeros((1, 10))
        
        # Training data storage - initialize as empty numpy arrays
        self.X_train = np.array([], dtype=np.float32).reshape(0, 64)
        self.y_train = np.array([], dtype=np.float32).reshape(0, 10)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            if self.last_x and self.last_y:
                # Create smooth line with rounded ends
                self.canvas.create_line(
                    self.last_x, self.last_y, x, y,
                    width=15,
                    fill='white',
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                    smooth=True
                )
                
                # Add visual feedback with a subtle glow effect
                self.canvas.create_oval(
                    x-2, y-2, x+2, y+2,
                    fill='#4fc3f7',
                    outline='#4fc3f7'
                )
            
            self.last_x = x
            self.last_y = y
    
    def predict_digit(self):
        # Process the image and store it
        self.current_processed_image = self.preprocess_image()
        
        # Forward pass
        h1 = self.sigmoid(np.dot(self.current_processed_image, self.weights['h1']) + self.biases['h1'])
        output = self.softmax(np.dot(h1, self.weights['out']) + self.biases['out'])
        prediction = np.argmax(output)
        confidence = output[0][prediction] * 100
        
        # Update UI with prediction and confidence
        result_text = f'Predicted: {prediction} (Confidence: {confidence:.1f}%)'    
        self.result_label.configure(text=result_text, foreground='#2196F3')
        self.predict_btn.state(['!disabled'])
        
        # Provide voice feedback
        feedback_text = f"The predicted digit is {prediction}"
        self.engine.say(feedback_text)
        self.engine.runAndWait()
        
        # Show training frame for feedback
        self.training_frame.grid()
        
        return prediction
    
    def clear_canvas(self):
        self.canvas.delete('all')
        self.result_label.configure(text='Draw a digit above', foreground='#2196F3')
        self.training_frame.grid_remove()
        
        # Add subtle animation effect
        self.canvas.create_text(
            140, 140,
            text='✏️',
            font=('Helvetica', 24),
            fill='#4fc3f7'
        )
        self.root.after(500, lambda: self.canvas.delete('all'))
    
    def stop_drawing(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        self.canvas.delete('all')
        self.result_label.config(text='Draw a digit')
    
    def preprocess_image(self):
        # Process the canvas content directly
        img_array = np.zeros((280, 280), dtype=np.uint8)
        for item in self.canvas.find_all():
            if self.canvas.type(item) == 'line':
                coords = self.canvas.coords(item)
                x1, y1, x2, y2 = map(int, coords)
                cv2.line(img_array, (x1, y1), (x2, y2), (255,), thickness=15)
        
        # Resize and normalize
        img_array = cv2.resize(img_array, (8, 8), interpolation=cv2.INTER_LANCZOS4)
        img_array = img_array.reshape(1, 64).astype('float32') / 255.0
        
        return img_array
    
    def manual_train(self):
        try:
            # Get the correct digit from spinbox
            correct_digit = int(self.correct_digit.get())
            if not 0 <= correct_digit <= 9:
                raise ValueError("Digit must be between 0 and 9")
            
            # Save the training sample
            self.training_handler.save_training_sample(self.current_processed_image, correct_digit)
            
            # Create one-hot encoded target
            target = np.zeros((1, 10))
            target[0, correct_digit] = 1
            
            # Add to training data using numpy concatenate
            self.X_train = np.concatenate([self.X_train, self.current_processed_image], axis=0) if self.X_train.size > 0 else self.current_processed_image
            self.y_train = np.concatenate([self.y_train, target], axis=0) if self.y_train.size > 0 else target
            
            # Use the arrays directly
            X = self.X_train
            y = self.y_train
            
            # Training with backpropagation
            learning_rate = 0.1
            
            # Forward pass
            hidden = self.sigmoid(np.dot(X, self.weights['h1']) + self.biases['h1'])
            output = self.softmax(np.dot(hidden, self.weights['out']) + self.biases['out'])
            
            # Backward pass
            output_error = output - y
            hidden_error = np.dot(output_error, self.weights['out'].T) * self.sigmoid_derivative(hidden)
            
            # Update weights and biases
            self.weights['out'] -= learning_rate * np.dot(hidden.T, output_error)
            self.biases['out'] -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
            self.weights['h1'] -= learning_rate * np.dot(X.T, hidden_error)
            self.biases['h1'] -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True)
            
            # Save model
            model_data = {
                'weights': self.weights,
                'biases': self.biases
            }
            self.training_handler.save_model(model_data)
            
            # Update UI
            self.result_label.configure(
                text='Training successful!',
                foreground='#4caf50'
            )
            self.training_frame.grid_remove()
            
        except Exception as e:
            self.result_label.configure(
                text=f'Training error: {str(e)}',
                foreground='#f44336'
            )

if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()