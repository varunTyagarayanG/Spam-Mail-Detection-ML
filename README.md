# Machine Learning Code Execution Guide

## Installations

You'll need to set up a Python environment and install libraries commonly used in machine learning. 

You don't need all of them to run this file. The following command is all you need:

```bash
pip install numpy pandas scikit-learn
```

# Algo! :
## Step-by-Step Algorithm for Running the ML Model

### 1. Import Libraries
- Import necessary libraries for data manipulation, machine learning, and evaluation.

### 2. Load the Dataset
- Load the dataset from a CSV file into a pandas DataFrame.

### 3. Handle Missing Values
- Replace null values in the DataFrame with empty strings.

### 4. Encode Labels
- Encode the 'Category' column labels: 'spam' as 0 and 'ham' as 1.

### 5. Separate Features and Labels
- Separate the feature (messages) and the target (labels).

### 6. Split the Data
- Split the data into training and test sets.

### 7. Transform Text Data to Feature Vectors
- Transform the text data into TF-IDF feature vectors.

### 8. Train the Logistic Regression Model
- Train a logistic regression model using the training data.

### 9. Make Predictions
- Use the trained model to make predictions on the test data.

### 10. Evaluate the Model
- Evaluate the model's performance using a confusion matrix and accuracy score.

# Output : 

## 1 .Confusion Matrix
       147  49
        0   1197
        
        True Negatives (TN)  : 147
        False Positives (FP) : 49
        False Negatives (FN) : 0
        True Positives (TP)  : 1197
## 2 .Accuracy Score
    
    0.964824120603015
    
An accuracy score of 96.48% means that the model correctly predicted the category of emails about 96.48% of the time.

