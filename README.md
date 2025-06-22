# Week 3 Python Assignment: Machine Learning Tasks

This repository contains implementations for two machine learning tasks: Classical ML with Scikit-learn and Deep Learning with TensorFlow.

## 📋 Overview

### Task 1: Classical ML with Scikit-learn
- **Dataset**: Iris Species Dataset
- **Algorithm**: Decision Tree Classifier
- **Goal**: Preprocess data, train classifier, evaluate with accuracy/precision/recall
- **Result**: ✅ **93.3% accuracy achieved**

### Task 2: Deep Learning with TensorFlow
- **Dataset**: MNIST Handwritten Digits
- **Algorithm**: Convolutional Neural Network (CNN)
- **Goal**: Build CNN model, achieve >95% test accuracy, visualize predictions
- **Result**: ✅ **99.31% accuracy achieved** (exceeded goal!)

## 🚀 Quick Start

### Prerequisites
Install the required dependencies:

```bash
# For Task 1 (Classical ML)
pip install pandas scikit-learn numpy

# For Task 2 (Deep Learning)
pip install -r requirements_deep_learning.txt
```

### Running the Tasks

#### Task 1: Iris Classification
```bash
python iris_decision_tree.py
```

#### Task 2: MNIST CNN Classification
```bash
python mnist_cnn_classifier.py
```

## 📊 Results Summary

### Task 1: Iris Decision Tree Classifier
- **Accuracy**: 93.3% (28/30 correct predictions on test set)
- **Precision**: 93.3% (weighted average)
- **Recall**: 93.3% (weighted average)
- **Key Insights**:
  - Petal measurements are most discriminative features
  - PetalLengthCm: 55.9% importance
  - PetalWidthCm: 40.6% importance
  - Tree depth: 5 levels, 8 leaves

### Task 2: MNIST CNN Classifier
- **Accuracy**: 99.31% (9,931/10,000 correct predictions)
- **Test Loss**: 0.0263
- **Precision**: 99.31% (weighted average)
- **Recall**: 99.31% (weighted average)
- **Training Time**: ~12 epochs (early stopping)
- **Model Size**: 93,322 parameters (364.54 KB)

## 🏗️ Model Architectures

### Task 1: Decision Tree
```
Decision Tree Classifier
├── Features: 4 (SepalLength, SepalWidth, PetalLength, PetalWidth)
├── Classes: 3 (Iris-setosa, Iris-versicolor, Iris-virginica)
├── Max Depth: 5
└── Leaves: 8
```

### Task 2: CNN Architecture
```
Sequential CNN Model
├── Conv2D(32, 3x3) + ReLU + MaxPool2D(2x2)
├── Conv2D(64, 3x3) + ReLU + MaxPool2D(2x2)  
├── Conv2D(64, 3x3) + ReLU
├── Flatten
├── Dense(64) + ReLU + Dropout(0.5)
└── Dense(10) + Softmax
```

## 📁 File Structure

```
week3_Python_assignment/
├── iris_decision_tree.py              # Task 1: Classical ML script
├── mnist_cnn_classifier.py            # Task 2: Deep learning script
├── Iris.csv                           # Iris dataset
├── mnist_cnn_model.h5                 # Trained CNN model
├── training_history.png               # Training curves visualization
├── mnist_predictions_visualization.png # Sample predictions visualization
├── requirements_deep_learning.txt     # Deep learning dependencies
└── README.md                          # This documentation
```

## 🔍 Detailed Implementation

### Task 1: Classical ML Pipeline
1. **Data Loading**: Load Iris dataset from CSV
2. **Preprocessing**: 
   - Drop ID column
   - Handle missing values with mean imputation
   - Encode species labels with LabelEncoder
3. **Train/Test Split**: 80/20 stratified split
4. **Model Training**: DecisionTreeClassifier with default parameters
5. **Evaluation**: Calculate accuracy, precision, recall
6. **Feature Analysis**: Analyze feature importance

### Task 2: Deep Learning Pipeline
1. **Data Loading**: Load MNIST from Keras datasets
2. **Preprocessing**:
   - Normalize pixel values to [0,1]
   - Reshape for CNN input (28x28x1)
   - One-hot encode labels
3. **Model Architecture**: 3-layer CNN with dropout
4. **Training**: 
   - Adam optimizer
   - Categorical crossentropy loss
   - Early stopping and learning rate reduction
5. **Evaluation**: Comprehensive metrics and classification report
6. **Visualization**: Plot 5 sample predictions with confidence scores

## 📈 Performance Analysis

### Task 1 Insights
- **Excellent Performance**: 93.3% accuracy on a 3-class problem
- **Feature Importance**: Petal measurements dominate (96.5% combined importance)
- **Model Interpretability**: Decision tree provides clear decision rules
- **Balanced Dataset**: Equal representation of all iris species

### Task 2 Insights
- **Outstanding Performance**: 99.31% accuracy exceeds 95% goal
- **Fast Convergence**: Model converged in 12 epochs
- **Robust Architecture**: CNN effectively captures spatial features
- **High Confidence**: Most predictions have >99% confidence
- **Generalization**: No overfitting observed

## 🛠️ Technical Details

### Dependencies
- **Task 1**: pandas, scikit-learn, numpy
- **Task 2**: tensorflow, keras, numpy, matplotlib, scikit-learn, seaborn

### Hardware Requirements
- **Task 1**: Minimal (runs on any modern CPU)
- **Task 2**: CPU sufficient (GPU optional for faster training)

### Training Time
- **Task 1**: <1 second
- **Task 2**: ~5 minutes on CPU

## 🎯 Goals Achievement

| Task | Goal | Result | Status |
|------|------|--------|--------|
| Task 1 | Preprocess data | ✅ Complete | Achieved |
| Task 1 | Train decision tree | ✅ 93.3% accuracy | Achieved |
| Task 1 | Evaluate metrics | ✅ All metrics calculated | Achieved |
| Task 2 | Build CNN model | ✅ 3-layer CNN | Achieved |
| Task 2 | >95% test accuracy | ✅ 99.31% accuracy | **Exceeded** |
| Task 2 | Visualize predictions | ✅ 5 sample images | Achieved |

## 🔬 Future Improvements

### Task 1 Enhancements
- Hyperparameter tuning (max_depth, min_samples_split)
- Cross-validation for robust evaluation
- Comparison with other algorithms (Random Forest, SVM)
- Feature engineering and selection

### Task 2 Enhancements
- Data augmentation for improved generalization
- Advanced architectures (ResNet, DenseNet)
- Ensemble methods
- Deployment optimization

## 📝 Usage Examples

### Loading Trained Models
```python
# Task 2: Load trained CNN model
from tensorflow import keras
model = keras.models.load_model('mnist_cnn_model.h5')

# Make predictions on new data
predictions = model.predict(new_images)
```

### Reproducing Results
Both scripts include random seed setting for reproducible results:
- Task 1: `random_state=42` in all sklearn functions
- Task 2: `tf.random.set_seed(42)` and `np.random.seed(42)`

## 📞 Contact & Support

For questions or issues with this implementation, please refer to the comprehensive comments in the Python scripts which include step-by-step explanations.

---

**Assignment Completion**: ✅ Both tasks successfully completed with excellent results! 