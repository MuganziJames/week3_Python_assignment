# Part 1: Theoretical Understanding (40%)

## 1. Short Answer Questions

### Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?

**Primary Differences:**

1. **Computational Graph Approach:**
   - **TensorFlow**: Uses static computational graphs (define-then-run). You first define the entire computation graph, then execute it in a session.
   - **PyTorch**: Uses dynamic computational graphs (define-by-run). The graph is built on-the-fly during execution, making it more intuitive and flexible.

2. **Ease of Debugging:**
   - **TensorFlow**: Debugging can be challenging due to static graphs; requires special debugging tools.
   - **PyTorch**: Easier to debug since you can use standard Python debugging tools like pdb, as the code executes line by line.

3. **Learning Curve:**
   - **TensorFlow**: Steeper learning curve, especially for beginners due to its complexity and multiple APIs.
   - **PyTorch**: More Pythonic and intuitive, easier for Python developers to pick up.

4. **Production Deployment:**
   - **TensorFlow**: Better ecosystem for production deployment with TensorFlow Serving, TensorFlow Lite, and TensorFlow.js.
   - **PyTorch**: Improving with TorchScript and TorchServe, but traditionally weaker in production tools.

**When to Choose:**

**Choose TensorFlow when:**
- You need robust production deployment capabilities
- Working on mobile or web deployment (TensorFlow Lite, TensorFlow.js)
- You prefer a more structured, enterprise-ready framework
- Working with large-scale distributed training

**Choose PyTorch when:**
- You're doing research or prototyping
- You want more flexibility and dynamic model building
- You prefer a more intuitive, Pythonic approach
- You need easier debugging capabilities
- Working on computer vision or NLP research projects

### Q2: Describe two use cases for Jupyter Notebooks in AI development.

**Use Case 1: Exploratory Data Analysis (EDA) and Data Preprocessing**

Jupyter Notebooks excel in data exploration and preprocessing phases of AI projects:
- **Interactive visualization**: Create plots, charts, and graphs to understand data distributions, correlations, and patterns
- **Iterative analysis**: Test different data cleaning approaches, feature engineering techniques, and statistical analyses
- **Documentation**: Combine code, visualizations, and markdown explanations in a single document
- **Example**: Analyzing a dataset of customer transactions to identify patterns, outliers, and relationships before building a recommendation system

**Use Case 2: Model Prototyping and Experimentation**

Jupyter Notebooks are ideal for rapid model development and comparison:
- **Quick iterations**: Test different algorithms, hyperparameters, and architectures in separate cells
- **Results comparison**: Display model performance metrics, confusion matrices, and validation curves side-by-side
- **Collaborative research**: Share notebooks with team members to demonstrate findings and methodologies
- **Example**: Comparing different neural network architectures for image classification, testing various optimizers, and visualizing training progress in real-time

### Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?

**spaCy Enhancements:**

1. **Linguistic Intelligence:**
   - **Basic Python**: Only handles text as sequences of characters
   - **spaCy**: Understands linguistic structures (sentences, tokens, parts of speech, syntactic dependencies)

2. **Advanced Tokenization:**
   - **Basic Python**: `split()` method breaks on whitespace, missing punctuation handling
   - **spaCy**: Intelligent tokenization that handles contractions, punctuation, and language-specific rules

3. **Named Entity Recognition (NER):**
   - **Basic Python**: No built-in capability to identify entities
   - **spaCy**: Automatically identifies and classifies entities (persons, organizations, locations, dates)

4. **Part-of-Speech Tagging:**
   - **Basic Python**: Cannot determine grammatical roles of words
   - **spaCy**: Provides detailed POS tags and morphological analysis

5. **Semantic Understanding:**
   - **Basic Python**: No understanding of word relationships or meanings
   - **spaCy**: Offers word vectors, similarity calculations, and semantic relationships

6. **Language Models:**
   - **Basic Python**: No pre-trained models
   - **spaCy**: Comes with pre-trained models for multiple languages with statistical accuracy

**Example Comparison:**
```python
# Basic Python approach
text = "Apple Inc. is looking at buying U.K. startup for $1 billion"
words = text.split()  # Simple splitting

# spaCy approach
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)  # Apple Inc. - ORG, U.K. - GPE, $1 billion - MONEY
```

## 2. Comparative Analysis: Scikit-learn vs TensorFlow

### Target Applications

**Scikit-learn:**
- **Classical Machine Learning**: Linear regression, logistic regression, SVM, decision trees, random forests
- **Traditional algorithms**: K-means clustering, PCA, feature selection
- **Small to medium datasets**: Typically works well with datasets that fit in memory
- **Structured data**: Excels with tabular data, feature matrices
- **Quick prototyping**: Rapid implementation of standard ML algorithms

**TensorFlow:**
- **Deep Learning**: Neural networks, CNNs, RNNs, Transformers
- **Large-scale problems**: Image recognition, natural language processing, speech recognition
- **Big data**: Designed for large datasets that may not fit in memory
- **Unstructured data**: Images, text, audio, video processing
- **Production deployment**: Scalable solutions for real-world applications

### Ease of Use for Beginners

**Scikit-learn:**
- **Highly beginner-friendly**: Consistent API across all algorithms
- **Simple workflow**: fit(), predict(), score() methods for all models
- **Extensive documentation**: Clear examples and tutorials
- **Minimal setup**: Easy installation and immediate use
- **Built-in utilities**: Preprocessing, model selection, and evaluation tools
- **Example**: 
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**TensorFlow:**
- **Steeper learning curve**: Requires understanding of neural networks and deep learning concepts
- **More complex setup**: GPU configuration, environment setup can be challenging
- **Multiple APIs**: Keras (high-level) vs Core TensorFlow (low-level) can be confusing
- **Debugging complexity**: Error messages can be cryptic for beginners
- **Requires more background**: Understanding of backpropagation, optimization, etc.

### Community Support

**Scikit-learn:**
- **Mature ecosystem**: Established since 2007, stable and well-tested
- **Strong documentation**: Comprehensive user guide with mathematical explanations
- **Active community**: Regular updates, bug fixes, and feature additions
- **Integration**: Works seamlessly with NumPy, Pandas, Matplotlib
- **Educational resources**: Widely used in academic courses and tutorials

**TensorFlow:**
- **Massive community**: Backed by Google, large developer base
- **Extensive resources**: TensorFlow Hub, Model Garden, extensive tutorials
- **Regular updates**: Frequent releases with new features and improvements
- **Industry adoption**: Used by major companies, ensuring continued development
- **Specialized communities**: Separate communities for different domains (computer vision, NLP)
- **Research integration**: Close ties with academic research and state-of-the-art models

**Conclusion:**
- **Scikit-learn** is ideal for beginners learning machine learning fundamentals and working with structured data
- **TensorFlow** is better suited for advanced users tackling complex deep learning problems and requiring production-ready solutions
- Both have strong community support, but serve different segments of the ML community

# Part 3: Ethics & Optimization (10%)

## 1. Ethical Considerations

### Potential Biases in MNIST and Amazon Reviews Models

**MNIST Model Biases:**

1. **Demographic Bias:**
   - **Issue**: MNIST dataset primarily contains handwriting samples from American Census Bureau employees and high school students, potentially biasing toward Western writing styles
   - **Impact**: Model may perform poorly on handwriting from different cultural backgrounds or age groups
   - **Example**: Difficulty recognizing digits written by elderly individuals or those from non-Western countries

2. **Representation Bias:**
   - **Issue**: Limited diversity in handwriting styles, pen types, and writing conditions
   - **Impact**: Poor performance on real-world scenarios (different paper types, writing instruments, lighting conditions)
   - **Example**: Model trained on clean, standardized digits may fail on handwritten forms with varying quality

3. **Temporal Bias:**
   - **Issue**: Dataset is decades old and may not reflect modern handwriting patterns
   - **Impact**: Reduced accuracy on contemporary handwriting styles influenced by digital devices
   - **Example**: People who primarily use digital devices may have different handwriting characteristics

**Amazon Reviews Model Biases:**

1. **Selection Bias:**
   - **Issue**: Only users who choose to write reviews are represented, excluding silent majority
   - **Impact**: Skewed toward extreme opinions (very positive or very negative)
   - **Example**: Moderate satisfaction levels are underrepresented

2. **Demographic and Cultural Bias:**
   - **Issue**: Reviews may be dominated by certain demographic groups or geographic regions
   - **Impact**: Sentiment analysis may not generalize across different populations
   - **Example**: Cultural differences in expressing satisfaction or criticism

3. **Temporal Bias:**
   - **Issue**: Review patterns and language evolve over time
   - **Impact**: Model trained on older reviews may misinterpret contemporary language patterns
   - **Example**: New slang, emojis, or expression styles not captured in training data

4. **Product Category Bias:**
   - **Issue**: Different product categories may have different review patterns
   - **Impact**: Model may perform better on some product types than others
   - **Example**: Electronics reviews vs. books reviews may have different sentiment expressions

### Mitigation Strategies Using TensorFlow Fairness Indicators and spaCy

**Using TensorFlow Fairness Indicators:**

1. **Bias Detection and Measurement:**
```python
# Example implementation for MNIST bias detection
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators

# Define demographic groups for analysis
eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='digit')],
    slicing_specs=[
        tfma.SlicingSpec(),  # Overall
        tfma.SlicingSpec(feature_keys=['age_group']),  # By age group
        tfma.SlicingSpec(feature_keys=['geographic_region']),  # By region
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                fairness_indicators.FairnessIndicators(
                    thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]
                )
            ]
        )
    ]
)
```

2. **Bias Mitigation Techniques:**
   - **Data Augmentation**: Generate synthetic samples to balance demographic representation
   - **Adversarial Debiasing**: Train model to be invariant to sensitive attributes
   - **Post-processing**: Adjust predictions to achieve fairness across groups

**Using spaCy's Rule-Based Systems:**

1. **Bias Detection in Text:**
```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define patterns for potentially biased language
bias_patterns = [
    [{"LOWER": {"IN": ["obviously", "clearly", "naturally"]}],  # Assumption markers
    [{"LOWER": "typical"}, {"POS": "NOUN"}],  # Stereotyping patterns
]

for pattern in bias_patterns:
    matcher.add("BIAS_PATTERN", [pattern])

def detect_bias_markers(text):
    doc = nlp(text)
    matches = matcher(doc)
    return [(doc[start:end].text, doc[start:end].label_) for match_id, start, end in matches]
```

2. **Demographic-Aware Preprocessing:**
```python
# Use spaCy to identify and handle demographic references
def preprocess_for_fairness(text):
    doc = nlp(text)
    
    # Remove or anonymize demographic identifiers
    filtered_tokens = []
    for token in doc:
        if token.ent_type_ in ["PERSON", "GPE", "ORG"]:
            filtered_tokens.append("[ENTITY]")  # Anonymize entities
        elif token.pos_ == "PROPN" and token.text.lower() in demographic_terms:
            filtered_tokens.append("[DEMOGRAPHIC]")  # Anonymize demographic terms
        else:
            filtered_tokens.append(token.text)
    
    return " ".join(filtered_tokens)
```

**Comprehensive Bias Mitigation Strategy:**

1. **Data Collection Phase:**
   - Ensure diverse, representative sampling
   - Document data sources and collection methods
   - Implement stratified sampling across demographic groups

2. **Model Development Phase:**
   - Use fairness-aware training objectives
   - Implement cross-validation across demographic groups
   - Regular bias auditing during development

3. **Evaluation Phase:**
   - Test performance across different demographic groups
   - Use multiple fairness metrics (equalized odds, demographic parity)
   - Conduct adversarial testing with edge cases

4. **Deployment Phase:**
   - Continuous monitoring for bias drift
   - Regular retraining with updated, diverse data
   - Feedback mechanisms for bias reporting

## 2. Troubleshooting Challenge

### Buggy TensorFlow Script Example and Fixes

Here's an example of a buggy TensorFlow script with common errors and their fixes:

**Buggy Code:**
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Buggy MNIST CNN with multiple errors
def buggy_mnist_model():
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # ERROR 1: Missing normalization and wrong reshape
    x_train = x_train.reshape(60000, 28, 28)  # Missing channel dimension
    x_test = x_test.reshape(10000, 28, 28)    # Missing channel dimension
    
    # ERROR 2: Wrong label encoding for categorical_crossentropy
    # Labels are kept as integers instead of one-hot encoding
    
    # ERROR 3: Model architecture issues
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28)),  # Wrong input shape
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='relu')  # Wrong activation for final layer
    ])
    
    # ERROR 4: Wrong loss function for the label format
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Wrong loss for integer labels
                  metrics=['accuracy'])
    
    # ERROR 5: Dimension mismatch in training
    history = model.fit(x_train, y_train,
                       batch_size=32,
                       epochs=5,
                       validation_data=(x_test, y_test))
    
    return model, history

# This code will produce multiple errors!
```

**Fixed Code with Explanations:**
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

def fixed_mnist_model():
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # FIX 1: Proper normalization and reshape with channel dimension
    x_train = x_train.astype('float32') / 255.0  # Normalize to [0,1]
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(60000, 28, 28, 1)  # Add channel dimension
    x_test = x_test.reshape(10000, 28, 28, 1)
    
    # FIX 2: Convert labels to categorical (one-hot) for categorical_crossentropy
    y_train_categorical = keras.utils.to_categorical(y_train, 10)
    y_test_categorical = keras.utils.to_categorical(y_test, 10)
    
    # FIX 3: Correct model architecture
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', 
                           input_shape=(28, 28, 1)),  # Correct input shape with channel
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')  # Softmax for multiclass classification
    ])
    
    # FIX 4: Correct loss function for categorical labels
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Matches one-hot encoded labels
                  metrics=['accuracy'])
    
    # Alternative approach: Use sparse_categorical_crossentropy with integer labels
    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy',  # For integer labels
    #               metrics=['accuracy'])
    
    # FIX 5: Use categorical labels for training
    history = model.fit(x_train, y_train_categorical,  # Use categorical labels
                       batch_size=32,
                       epochs=5,
                       validation_data=(x_test, y_test_categorical))
    
    return model, history

# This fixed version will work correctly!
```

**Common TensorFlow Error Types and Solutions:**

1. **Dimension Mismatch Errors:**
   - **Error**: `ValueError: Input 0 of layer sequential is incompatible with the layer`
   - **Solution**: Check input shapes, add/remove dimensions as needed with `reshape()` or `expand_dims()`

2. **Loss Function Mismatch:**
   - **Error**: `InvalidArgumentError: logits and labels must have the same first dimension`
   - **Solution**: Match loss function to label format (categorical vs sparse categorical)

3. **Data Type Issues:**
   - **Error**: `TypeError: Input 'y' of 'Sub' Op has type float32 that does not match expected type of int32`
   - **Solution**: Ensure consistent data types using `astype()` conversions

4. **Memory Issues:**
   - **Error**: `ResourceExhaustedError: OOM when allocating tensor`
   - **Solution**: Reduce batch size, use data generators, or optimize model architecture

**Debugging Best Practices:**

1. **Print Shapes**: Always print tensor shapes during development
2. **Start Simple**: Begin with minimal models and gradually add complexity
3. **Use Model Summary**: Call `model.summary()` to verify architecture
4. **Check Data**: Validate input data ranges, types, and shapes
5. **Test Small Batches**: Train on small batches first to catch errors quickly
