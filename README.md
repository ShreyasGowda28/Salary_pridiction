# Salary Prediction App

This project is a Flask web application that predicts the salary of an employee based on various input features such as age, gender, designation, department, past experience, and performance ratings. The model also provides recommendations based on the total experience of the employee.

## upervised by

Prof. Agughasi Victor Ikechukwu, (Assistant Professor) Department of CSE, MIT Mysore)

## Collaborators

4MH21CS028 Gokulnath UC

4MH21CS074 R Bilwananda

4MH21CS086 Shalom Raj J

4MH21CS092 Shreyas Gowda S

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview
The application utilizes a pre-trained machine learning model to predict the salary of an employee. The model is built using the RandomForestRegressor algorithm and trained on a dataset containing various employee attributes and their corresponding salaries.

## Installation
To get started with the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ShreyasGowda28/salary-prediction-app.git
   cd salary-prediction-app
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**
   ```bash
   flask run
   ```

## Usage
1. Open your web browser and navigate to `http://127.0.0.1:5000/`.
2. Fill in the form with the required employee details.
3. Submit the form to get the predicted salary and recommendations.

## Model Training
The model is trained using the `RandomForestRegressor` from scikit-learn. Below are the steps for training the model:

## Load and Preprocess the Dataset
```python
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
with open('salary_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to preprocess input data
def preprocess_input(data):
    # Convert 'DOJ' to datetime
    data['DOJ'] = pd.to_datetime(data['DOJ'])

    # Hardcoded current date for example, in real scenario use datetime.now()
    data['CURRENT DATE'] = pd.to_datetime('2016-07-01')

    # Calculate tenure in days, then convert to years
    data['TENURE'] = (data['CURRENT DATE'] - data['DOJ']).dt.days // 365.25
    data['TOTAL_EXP'] = data['PAST_EXP'] + data['TENURE']

    # Drop unnecessary columns
    data = data.drop(columns=['DOJ', 'CURRENT DATE', 'PAST_EXP', 'TENURE', 'AGE'])

    return data

# Function to generate recommendation based on total experience
def generate_recommendation(total_experience):
    if total_experience >= 5:
        recommendation = "Your performance and experience suggest that you're well-positioned for a salary increase. Consider discussing this with your manager during your next performance review."
    elif total_experience >= 3:
        recommendation = "You've gained valuable experience and have a solid performance rating. It might be a good time to explore opportunities for advancement within the company or discuss a salary review with your manager."
    else:
        recommendation = "Focus on enhancing your skills, gaining more experience, and improving your performance to increase your chances of a salary raise in the future."
    return recommendation

# Route to render HTML form
@app.route('/')
def form():
    return render_template('form.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        name = request.form['name']
        age = int(request.form['age'])
        gender = request.form['gender']
        designation = request.form['designation']
        unit = request.form['unit']
        past_experience = float(request.form['past_experience'])
        rating = float(request.form['rating'])
        date_of_join = request.form['date_of_join']

        # Create a DataFrame with input data
        input_data = pd.DataFrame({
            'NAME': [name],
            'AGE': [age],
            'SEX': [gender],
            'DESIGNATION': [designation],
            'UNIT': [unit],
            'PAST_EXP': [past_experience],
            'RATINGS': [rating],
            'DOJ': [date_of_join]
        })

        # Preprocess input data
        preprocessed_data = preprocess_input(input_data)

        # Calculate total experience for recommendation
        total_experience = preprocessed_data['TOTAL_EXP'].iloc[0]

        # Generate recommendation
        recommendation = generate_recommendation(total_experience)

        # Predict using the model
        salary_prediction = model.predict(preprocessed_data)

        # Render result template with prediction and recommendation
        return render_template('result.html', name=name, prediction=salary_prediction[0], recommendation=recommendation)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)

```

## Split the Data and Define the Pipeline

```python
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
df = pd.read_csv("C:/Users/sudarshan uc/OneDrive/Desktop/Salary Prediction of Data Professions.csv")
df.info()
plt.figure(figsize=(10, 5))
sns.histplot(df['SALARY'], kde=True)
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()
print()
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['SALARY'])
plt.title('Box Plot of Salary')
plt.xlabel('Salary')
plt.show()
categorical_features = ['SEX','DESIGNATION', 'UNIT']
for feature in categorical_features:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df[feature], y=df['SALARY'])
    plt.title(f'Average Salary by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Average Salary')
    plt.xticks(rotation=45)
    plt.show()
    print()
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Calculate correlation matrix for numerical columns
corr = df[numerical_columns].corr()

# Plot correlation heatmap
plt.figure(dpi=130)
sns.heatmap(corr, annot=True, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
# Convert 'DOJ' and 'CURRENT DATE' to datetime
df['DOJ'] = pd.to_datetime(df['DOJ'])
df['CURRENT DATE'] = pd.to_datetime(df['CURRENT DATE'])

# Calculate tenure in days, then convert to years
df['TENURE'] = (df['CURRENT DATE'] - df['DOJ']).dt.days // 365.25
df['TOTAL_EXP'] = df['PAST EXP'] + df['TENURE']


df = df.drop(columns=['DOJ', 'CURRENT DATE','FIRST NAME','LAST NAME','LEAVES REMAINING','PAST EXP','TENURE','AGE','LEAVES USED'])
df =df.dropna()
df = df.drop_duplicates()
df.info()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["SALARY"]),df["SALARY"], test_size=0.2, random_state=42)
s1 = ColumnTransformer([
    ('ord', OrdinalEncoder(categories=[['Analyst', 'Associate', 'Senior Analyst', 'Manager', 'Senior Manager', 'Director']]), ['DESIGNATION']),
    ('onehot', OneHotEncoder(), ['UNIT', 'SEX'])
], remainder='passthrough')

s3 = ColumnTransformer([
    ('Scale', StandardScaler(), slice(0,122))
], remainder="passthrough")
# List of models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors Regression": KNeighborsRegressor(),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Random Forest Regression": RandomForestRegressor(),
    "Gradient Boosting Regression": GradientBoostingRegressor()
}

# Fit and evaluate each model
for name, model in models.items():
    pipeline = Pipeline([
        ("s1", s1),
        ("s2", model),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} R2 Score: {r2}")
s1 = ColumnTransformer([
    ('ord', OrdinalEncoder(categories=[['Analyst', 'Associate', 'Senior Analyst', 'Manager', 'Senior Manager', 'Director']]), ['DESIGNATION']),
    ('onehot', OneHotEncoder(), ['UNIT', 'SEX'])
], remainder='passthrough')

# Define the RandomForestRegressor with the best hyperparameters
s2 = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=4
)

# Create a pipeline with preprocessing and the model
model = Pipeline([
    ("s1", s1),
    ("s2", s2),
])

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)
# Function to preprocess input data
def preprocess_input(df):
    # Preprocess input data here (e.g., encoding categorical variables, transforming date features)
    # Convert 'DOJ' and 'CURRENT DATE' to datetime
    df['DOJ'] = pd.to_datetime(df['DOJ'])
    df['CURRENT DATE'] = pd.to_datetime('2016-07-01')

    # Calculate tenure in days, then convert to years
    df['TENURE'] = (df['CURRENT DATE'] - df['DOJ']).dt.days // 365.25
    df['TOTAL_EXP'] = df['PAST_EXP'] + df['TENURE']
    df = df.drop(columns=['DOJ', 'CURRENT DATE', 'NAME', 'PAST_EXP', 'TENURE', 'AGE'])
    total_experience=int(df['TOTAL_EXP'][0]) 
    # Provide salary increase recommendations based on total experience and rating
    if total_experience >= 5 :
        recommendation = "Your performance and experience suggest that you're well-positioned for a salary increase. Consider discussing this with your manager during your next performance review."
    elif total_experience >= 3 :
        recommendation = "You've gained valuable experience and have a solid performance rating. It might be a good time to explore opportunities for advancement within the company or discuss a salary review with your manager."
    else:
        recommendation = "Focus on enhancing your skills, gaining more experience, and improving your performance to increase your chances of a salary raise in the future."

    return df,recommendation

# Function to make prediction
def predict_salary(data):
    preprocessed_data,rec = preprocess_input(data)

    salary = model.predict(preprocessed_data)
    return salary,rec

# Terminal app
def main():
    print("Salary Prediction App")

#     # Input fields
#     name = input('Name: ')
#     age = int(input('Age: '))
#     gender = input('Gender (M/F): ')
#     designation = input('Designation (Analyst, Associate, Senior Analyst, Manager, Senior Manager, Director): ')
#     unit = input('Unit (Finance, Web, IT, Operations, Management, Marketing): ')
#     past_experience = float(input('Past Experience: '))
#     rating = float(input('Rating (0.0 - 5.0): '))
#     date_of_join = input('Date of Join (YYYY-MM-DD): ')


    input_data = pd.DataFrame({
        'NAME': ["AI"],
        'AGE': [40],
        'SEX': ["M"],
        'DESIGNATION': ["Director"],
        'UNIT': ["IT"],
        'PAST_EXP': [15],
        'RATINGS': [5.0],
        'DOJ': ["2014-09-30"]
    })

    salary_prediction,rec = predict_salary(input_data)
    print(f'Predicted Salary: {salary_prediction[0]}')
    print(f'Recommendation: {rec}')

if __name__ == '__main__':
    main()

```

### HTML Templates

**form.html**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Salary Prediction Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            width: 400px; /* Adjust width as needed */
            max-width: 100%;
            text-align: center;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        p {
            color: #555;
            font-size: 18px;
            margin-bottom: 10px;
        }
        .predicted-salary {
            color: #5cb85c;
            font-size: 24px;
            font-weight: bold;
        }
        .recommendation {
            color: #333;
            font-size: 16px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Salary Prediction Result</h1>
        <p>Hello, {{ name }}!</p>
        <p class="predicted-salary">Predicted Salary: ${{ prediction }}</p>
        <p class="recommendation">{{ recommendation }}</p>
    </div>
</body>
</html>

```

**result.html**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Salary Prediction Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            width: 400px; /* Adjust width as needed */
            max-width: 100%;
            text-align: center;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        p {
            color: #555;
            font-size: 18px;
            margin-bottom: 10px;
        }
        .predicted-salary {
            color: #5cb85c;
            font-size: 24px;
            font-weight: bold;
        }
        .recommendation {
            color: #333;
            font-size: 16px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Salary Prediction Result</h1>
        <p>Hello, {{ name }}!</p>
        <p class="predicted-salary">Predicted Salary: ${{ prediction }}</p>
        <p class="recommendation">{{ recommendation }}</p>
    </div>
</body>
</html>

```


## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request with your changes. Make sure to follow the existing code style and include appropriate tests.

## Contact
For any inquiries or feedback, please contact:
- Gokulnath UC: gokulnathuc@gmail.com
- R Bilwananda: bilwananda30@gmail.com
- Shalom Raj J: shalom.raj2747@gmail.com
- Shreyas Gowda S: shreyasgowda9535@gmail.com

---
