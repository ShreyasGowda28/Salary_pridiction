from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load dataset
df = pd.read_csv("C:/Users/sudarshan uc/OneDrive/Desktop/DS project/Salary Prediction of Data Professions.csv")
# Convert 'DOJ' and 'CURRENT DATE' to datetime
df['DOJ'] = pd.to_datetime(df['DOJ'])
df['CURRENT DATE'] = pd.to_datetime(df['CURRENT DATE'])

# Calculate tenure in days, then convert to years
df['TENURE'] = (df['CURRENT DATE'] - df['DOJ']).dt.days // 365.25
df['TOTAL_EXP'] = df['PAST EXP'] + df['TENURE']

# Drop irrelevant columns
df = df.drop(columns=['DOJ', 'CURRENT DATE', 'FIRST NAME', 'LAST NAME', 'LEAVES REMAINING', 'PAST EXP', 'TENURE', 'AGE', 'LEAVES USED'])
df = df.dropna()
df = df.drop_duplicates()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["SALARY"]), df["SALARY"], test_size=0.2, random_state=42)

# Define preprocessing and model pipeline
s1 = ColumnTransformer([
    ('ord', OrdinalEncoder(categories=[['Analyst', 'Associate', 'Senior Analyst', 'Manager', 'Senior Manager', 'Director']]), ['DESIGNATION']),
    ('onehot', OneHotEncoder(), ['UNIT', 'SEX'])
], remainder='passthrough')

model = Pipeline([
    ("s1", s1),
    ("s2", RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=2, min_samples_leaf=4))
])

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Save the model
with open('salary_prediction_model.pkl', 'wb') as file:
    pickle.dump(model, file)
