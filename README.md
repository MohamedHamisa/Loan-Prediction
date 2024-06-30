


# Loan Prediction

## Overview

This project aims to build a machine learning model to predict loan eligibility for applicants based on various features such as gender, marital status, education, number of dependents, income, loan amount, credit history, and property area. The dataset is preprocessed to handle missing values, encoded to convert categorical features to numerical values, and scaled for better performance of the machine learning models. Various machine learning algorithms are applied to determine the best-performing model.

## Dataset

The dataset contains the following columns:

- **Loan_ID**: Unique Loan ID
- **Gender**: Gender of the applicant
- **Married**: Applicant's marital status
- **Dependents**: Number of dependents
- **Education**: Applicant's education level
- **Self_Employed**: Whether the applicant is self-employed
- **ApplicantIncome**: Applicant's income
- **CoapplicantIncome**: Coapplicant's income
- **LoanAmount**: Loan amount in thousands
- **Loan_Amount_Term**: Term of the loan in months
- **Credit_History**: Credit history meets guidelines
- **Property_Area**: Area of property
- **Loan_Status**: Whether the loan was approved or not (target variable)

## Requirements

Ensure you have the following libraries installed:

```
pip install pandas numpy matplotlib seaborn scikit-learn termcolor pickle
```

## Data Preprocessing

1. **Handling Missing Values**:
    - Numerical columns: Missing values are imputed with the median.
    - Categorical columns: Missing values are imputed with the most frequent value (mode).

2. **Encoding Categorical Variables**:
    - Categorical variables are encoded into numerical values using label encoding.

3. **Scaling Features**:
    - Features are scaled using MinMaxScaler for better model performance.

## Data Visualization

Visualizations are used to understand the dataset better:
- **Distribution of Numerical Features**: Histograms with KDE plots.
- **Count Plots for Categorical Features**: Bar plots for categorical features.
- **Correlation Heatmap**: Heatmap to show correlation between features.
- **Pair Plot**: Pair plots to visualize relationships between features colored by loan status.
- **Box Plots**: Box plots to identify potential outliers.

## Models Used

The following machine learning models are trained and evaluated:

- Decision Tree Classifier
- Random Forest Classifier
- GaussianNB
- BernoulliNB
- Logistic Regression
- Ridge Classifier CV
- K-Nearest Neighbors (KNN)

## Model Evaluation

The models are evaluated based on their training and testing accuracy. The best-performing model is saved using `pickle` for future use.

## Code

Here is a summary of the code used in the project:

```
# Importing necessary libraries for warning handling
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
import pickle
from termcolor import colored

# Suppressing warnings for cleaner output
warnings.filterwarnings('ignore')

# Loading the dataset
loan_data = pd.read_csv('loan_data.csv')

# Handling non-numeric values
loan_data['Dependents'].replace('3+', 3, inplace=True)
loan_data['Dependents'] = loan_data['Dependents'].astype(float)

# Columns with numerical values
num_columns = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Dependents']
num_imputer = SimpleImputer(strategy='median')
loan_data[num_columns] = num_imputer.fit_transform(loan_data[num_columns])

# Columns with categorical values
cat_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
cat_imputer = SimpleImputer(strategy='most_frequent')
loan_data[cat_columns] = cat_imputer.fit_transform(loan_data[cat_columns])

# Label encoding for categorical variables
loan_data['Gender'] = loan_data['Gender'].map({'Male': 1, 'Female': 0})
loan_data['Married'] = loan_data['Married'].map({'Yes': 1, 'No': 0})
loan_data['Education'] = loan_data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
loan_data['Self_Employed'] = loan_data['Self_Employed'].map({'Yes': 1, 'No': 0})
loan_data['Property_Area'] = loan_data['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
loan_data['Loan_Status'] = loan_data['Loan_Status'].map({'Y': 1, 'N': 0})

# Distribution of Numerical Features
plt.figure(figsize=(12, 10))
for i, col in enumerate(num_columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(loan_data[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Count Plots for Categorical Features
plt.figure(figsize=(12, 10))
for i, col in enumerate(cat_columns, 1):
    plt.subplot(3, 2, i)
    sns.countplot(data=loan_data, x=col)
    plt.title(f'Count Plot of {col}')
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = loan_data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Pair Plot
sns.pairplot(loan_data, hue='Loan_Status')
plt.show()

# Box Plots to Identify Outliers
plt.figure(figsize=(12, 10))
for i, col in enumerate(num_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=loan_data, y=col)
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()

# Splitting the data into features and target variable
X = loan_data.drop(columns=['Loan_ID', 'Loan_Status'])
y = loan_data['Loan_Status']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing scalers
scaler = MinMaxScaler()

# Scaling the features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initializing models
models = {
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "GaussianNB": GaussianNB(),
    "BernoulliNB": BernoulliNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Ridge Classifier CV": RidgeClassifierCV(),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier()
}

# Training and evaluating models
model_train_scores = {}
model_test_scores = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    model_train_scores[model_name] = train_accuracy
    model_test_scores[model_name] = test_accuracy
    print(colored(f"{model_name} Training Score: {train_accuracy:.2f}", "green"))
    print(colored(f"{model_name} Testing Score: {test_accuracy:.2f}", "green"))

# Saving the best model using pickle
best_model = models["Random Forest Classifier"]  # Example, choose the best performing model
filename = 'best_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(best_model, file)

print(colored("The best model was successfully saved to disk.", "green", attrs=['reverse']))
```

## How to Run the Project

1. Ensure you have the required libraries installed.
2. Load the dataset into a pandas DataFrame.
3. Preprocess the data by handling missing values, encoding categorical variables, and scaling features.
4. Visualize the data to understand the distributions and relationships.
5. Train various machine learning models.
6. Evaluate the models and choose the best one.
7. Save the best model using `pickle`.

## Conclusion

This project demonstrates the process of building a machine learning model to predict loan eligibility. It involves data preprocessing, visualization, model training, evaluation, and saving the best model for future use.

---

