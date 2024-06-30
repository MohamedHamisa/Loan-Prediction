%pip install seaborn

# Importing necessary libraries for warning handling
import warnings

# Importing third-party libraries for data manipulation, numerical operations, and visualization
import pandas as pd  # For data manipulation and CSV file I/O
import numpy as np  # For numerical operations and mathematical functions
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For statistical graphics

# Importing libraries from scikit-learn for machine learning tasks
from sklearn.model_selection import train_test_split  # For data splitting (training & testing)
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For feature standardization and normalization
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation metrics
from sklearn.impute import SimpleImputer  # For handling missing values

# Importing library for colored text output
from termcolor import colored

# Suppressing warnings for cleaner output
warnings.filterwarnings('ignore')

# Printing a success message in green text with reverse attribute
print(colored("The required libraries were successfully imported...", "red", attrs=['reverse']))

# Loading the dataset
loan_data = pd.read_csv('loan_data.csv')

# Displaying the first few rows of the dataset
print(loan_data.head())

# Displaying basic information about the dataset
loan_data.info()

# Checking for missing values
missing_values = loan_data.isnull().sum()
print(missing_values)

# Handling non-numeric values
loan_data['Dependents'].replace('3+', 3, inplace=True)
loan_data['Dependents'] = loan_data['Dependents'].astype(float)

# Columns with numerical values
num_columns = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Dependents']

# Imputing missing values in numerical columns with median
num_imputer = SimpleImputer(strategy='median')
loan_data[num_columns] = num_imputer.fit_transform(loan_data[num_columns])

# Columns with categorical values
cat_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

# Imputing missing values in categorical columns with mode
cat_imputer = SimpleImputer(strategy='most_frequent')
loan_data[cat_columns] = cat_imputer.fit_transform(loan_data[cat_columns])

# Verifying that there are no more missing values
print(loan_data.isnull().sum())

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

# Importing machine learning models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier

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
import pickle

best_model = models["Random Forest Classifier"]  # Example, choose the best performing model
filename = 'best_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(best_model, file)

print(colored("The best model was successfully saved to disk.", "red", attrs=['reverse']))
