# mnist_cnn_classifier.py
"""
Deep Learning CNN Model for MNIST Handwritten Digit Classification

Goal: Build a CNN model to classify handwritten digits with >95% test accuracy
and visualize predictions on sample images.

Usage:
    python mnist_cnn_classifier.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import seaborn as sns


def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) preprocessed data
    """
    print("Loading MNIST dataset...")
    
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Normalize pixel values to [0, 1] range
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data to add channel dimension (for CNN)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Convert labels to categorical (one-hot encoding)
    y_train_categorical = keras.utils.to_categorical(y_train, 10)
    y_test_categorical = keras.utils.to_categorical(y_test, 10)
    
    print(f"Preprocessed training data shape: {x_train.shape}")
    print(f"Preprocessed test data shape: {x_test.shape}")
    print(f"Label range: {np.min(y_train)} to {np.max(y_train)}")
    
    return x_train, y_train, x_test, y_test, y_train_categorical, y_test_categorical


def build_cnn_model():
    """
    Build a Convolutional Neural Network model for MNIST classification.
    
    Returns:
        Compiled Keras model
    """
    print("\nBuilding CNN model architecture...")
    
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Regularization to prevent overfitting
        layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    return model


def train_model(model, x_train, y_train_categorical, x_test, y_test_categorical):
    """
    Train the CNN model with validation.
    
    Args:
        model: Compiled Keras model
        x_train, y_train_categorical: Training data and labels
        x_test, y_test_categorical: Test data and labels for validation
    
    Returns:
        Training history
    """
    print("\nStarting model training...")
    
    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=0.001
    )
    
    # Train the model
    history = model.fit(
        x_train, y_train_categorical,
        batch_size=128,
        epochs=15,
        validation_data=(x_test, y_test_categorical),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("Training completed!")
    return history


def evaluate_model(model, x_test, y_test, y_test_categorical):
    """
    Evaluate the trained model and display metrics.
    
    Args:
        model: Trained Keras model
        x_test, y_test: Test data and original labels
        y_test_categorical: Test labels in categorical format
    
    Returns:
        Test accuracy
    """
    print("\nEvaluating model performance...")
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical, verbose=0)
    
    # Make predictions
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate additional metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print("="*60)
    print("CNN MODEL PERFORMANCE ON MNIST TEST SET")
    print("="*60)
    print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Precision:      {precision:.4f} (weighted)")
    print(f"Recall:         {recall:.4f} (weighted)")
    print("="*60)
    
    # Check if we achieved the goal
    if test_accuracy > 0.95:
        print("🎉 SUCCESS: Model achieved >95% test accuracy!")
    else:
        print("⚠️  Model did not reach 95% test accuracy target.")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    return test_accuracy


def visualize_predictions(model, x_test, y_test, num_samples=5):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: Trained Keras model
        x_test, y_test: Test data and labels
        num_samples: Number of samples to visualize
    """
    print(f"\nVisualizing predictions on {num_samples} sample images...")
    
    # Select random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    sample_images = x_test[indices]
    sample_labels = y_test[indices]
    
    # Make predictions
    predictions = model.predict(sample_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    # Create visualization
    plt.figure(figsize=(15, 3))
    
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        
        # Display image
        plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
        
        # Create title with prediction info
        true_label = sample_labels[i]
        pred_label = predicted_labels[i]
        confidence = confidence_scores[i]
        
        # Color code: green for correct, red for incorrect
        color = 'green' if true_label == pred_label else 'red'
        
        plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}', 
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.suptitle('CNN Model Predictions on MNIST Test Samples', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('mnist_predictions_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed predictions
    print("\nDetailed Prediction Results:")
    print("-" * 50)
    for i in range(num_samples):
        true_label = sample_labels[i]
        pred_label = predicted_labels[i]
        confidence = confidence_scores[i]
        status = "✓ Correct" if true_label == pred_label else "✗ Incorrect"
        
        print(f"Sample {i+1}: True={true_label}, Predicted={pred_label}, "
              f"Confidence={confidence:.3f} - {status}")


def plot_training_history(history):
    """
    Plot training history (accuracy and loss curves).
    
    Args:
        history: Training history from model.fit()
    """
    print("\nPlotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to orchestrate the entire CNN training and evaluation process.
    """
    print("="*70)
    print("MNIST HANDWRITTEN DIGIT CLASSIFICATION WITH CNN")
    print("="*70)
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Step 1: Load and preprocess data
    x_train, y_train, x_test, y_test, y_train_cat, y_test_cat = load_and_preprocess_data()
    
    # Step 2: Build CNN model
    model = build_cnn_model()
    
    # Step 3: Train the model
    history = train_model(model, x_train, y_train_cat, x_test, y_test_cat)
    
    # Step 4: Evaluate the model
    test_accuracy = evaluate_model(model, x_test, y_test, y_test_cat)
    
    # Step 5: Visualize predictions
    visualize_predictions(model, x_test, y_test, num_samples=5)
    
    # Step 6: Plot training history
    plot_training_history(history)
    
    # Step 7: Save the model
    model.save('mnist_cnn_model.h5')
    print(f"\nModel saved as 'mnist_cnn_model.h5'")
    
    print("\n" + "="*70)
    print("TASK COMPLETION SUMMARY")
    print("="*70)
    print(f"✓ CNN model built with proper architecture")
    print(f"✓ Model trained and evaluated")
    print(f"✓ Test accuracy achieved: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"✓ Goal of >95% accuracy: {'ACHIEVED' if test_accuracy > 0.95 else 'NOT ACHIEVED'}")
    print(f"✓ Predictions visualized on 5 sample images")
    print(f"✓ Training history plotted and saved")
    print(f"✓ Model saved for future use")
    print("="*70)


if __name__ == "__main__":
    main() 