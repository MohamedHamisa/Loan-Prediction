import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

# Load the dataset (replace with correct local path or valid direct download link)
# df = pd.read_csv("path_to_your_dataset.csv")
# For example:
# df = pd.read_csv("loan_dataset.csv")

# Example code with corrected data loading
try:
    df = pd.read_csv("loan_dataset.csv")  # Replace with your dataset path
except FileNotFoundError:
    print("Dataset file not found. Please provide the correct path or URL.")

# Check loaded data
print(df.head())
print(df.info())

# Data cleaning and preprocessing
# Fill missing values
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mean(), inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

# Data visualization
plt.figure(figsize=(10, 6))
sns.countplot(df['Gender'])
plt.title('Distribution of Gender')
plt.show()

plt.figure(figsize=(10, 6))
sns.distplot(df['ApplicantIncome'])
plt.title('Distribution of Applicant Income')
plt.show()

# Convert categorical to numerical
cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status', 'Dependents']
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

# Split data into train and test sets
X = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define function to classify and evaluate models
def classify(model, x, y):
    model.fit(x_train, y_train)
    print("Accuracy is", model.score(x_test, y_test) * 100)
    score = cross_val_score(model, x, y, cv=5)
    print("Cross validation is", np.mean(score) * 100)

# Example usage
model = LogisticRegression()
classify(model, X, y)

model = DecisionTreeClassifier()
classify(model, X, y)

model = RandomForestClassifier()
classify(model, X, y)

model = ExtraTreesClassifier()
classify(model, X, y)

# Fine-tune Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=1)
classify(model, X, y)

# Fit Random Forest Classifier on the training data
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate model using confusion matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
