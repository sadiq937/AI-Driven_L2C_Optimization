import numpy as np
import pandas as pd

# Load the dataset (Assuming you downloaded it and the file is named 'crm_sales_opportunities.csv')
df = pd.read_csv('crm_sales_opportunities.csv')

# Display first few rows of the dataset
print(df.head())

# You might need to clean and preprocess the data depending on the dataset (e.g., handling missing values)
# Handle missing data (e.g., filling missing values or dropping rows/columns)
df = df.dropna()  # Example: Dropping rows with missing values

# Feature selection - Identify relevant features (e.g., demographics, firmographics, behavioral data)
features = ['age', 'industry', 'company_size', 'website_visits', 'email_interactions', 'previous_purchases']  # Example columns

# Target variable (Assuming we want to predict 'converted' column)
target = 'converted'  # 1: Converted, 0: Not Converted

X = df[features]
y = df[target]

# If any categorical data exists, encode it
X = pd.get_dummies(X, drop_first=True)  # Converts categorical features to numerical via one-hot encoding
