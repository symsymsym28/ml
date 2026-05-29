import pandas as pd
from sklearn import datasets

# Load Dataset
df = pd.read_csv('Fraud_Detection_Transactions_Dataset.csv')

# Basic Operations
mean = df['Account_Balance'].mean()
median = df['Transaction_Amount'].median()
mode = df['Transaction_Distance'].mode()
variance = df['Risk_Score'].var()
standard_deviation = df['Daily_Transaction_Count'].std()

# Filtering Data
df_filtered = df[df['Authentication_Method'] == 'Biometric']

# Handling Missing Values
df['Card_Age'] = df['Card_Age'].fillna(df['Card_Age'].mean())

# Feature Normalization (Min-Max)
amount_min = df['Transaction_Amount'].min()
amount_max = df['Transaction_Amount'].max()
df['Transaction_Amount_Normalized'] = (df['Transaction_Amount'] - amount_min) / (amount_max - amount_min)