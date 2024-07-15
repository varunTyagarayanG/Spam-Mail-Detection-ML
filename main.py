import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Replace the null values with empty strings
mail_data = pd.read_csv('mail_data.csv')
Changed_mail_data = mail_data.where(pd.notnull(mail_data), '')

# Encode labels: spam = 0, ham = 1
Changed_mail_data.loc[Changed_mail_data['Category'] == 'spam', 'Category'] = 0
Changed_mail_data.loc[Changed_mail_data['Category'] == 'ham', 'Category'] = 1 

X = Changed_mail_data['Message']
Y = Changed_mail_data['Category'].astype(int)

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=3)

# Transform text data to feature vectors using TF-IDF
vectorizer = TfidfVectorizer(min_df=1 , stop_words='english',lowercase=True)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the logistic regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
cm = confusion_matrix(Y_test, y_pred)
print(cm)
print(accuracy_score(Y_test, y_pred))
