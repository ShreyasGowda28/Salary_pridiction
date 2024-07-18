Salary Prediction App

This project is a Flask web application that predicts the salary of an employee based on various input features such as age, gender, designation, department, past experience, and performance ratings. The model also provides recommendations based on the total experience of the employee.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview
The application utilizes a pre-trained machine learning model to predict the salary of an employee. The model is built using the RandomForestRegressor algorithm and trained on a dataset containing various employee attributes and their corresponding salaries.

## Installation
To get started with the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/salary-prediction-app.git
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

### Load and Preprocess the Dataset
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("path_to_dataset/Salary Prediction of Data Professions.csv")

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
```

### Split the Data and Define the Pipeline
```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

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

# Save the model
import pickle
with open('salary_prediction_model.pkl', 'wb') as file:
    pickle.dump(model, file)
```

## Web Application
The web application is built using Flask. Below are the key parts of the Flask application:

### Application Setup and Routes
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
    data['DOJ'] = pd.to_datetime(data['DOJ'])
    data['CURRENT DATE'] = pd.to_datetime('2016-07-01')
    data['TENURE'] = (data['CURRENT DATE'] - data['DOJ']).dt.days // 365.25
    data['TOTAL_EXP'] = data['PAST_EXP'] + data['TENURE']
    data = data.drop(columns=['DOJ', 'CURRENT DATE', 'PAST_EXP', 'TENURE', 'AGE'])
    return data

# Function to generate recommendation based on total experience
def generate_recommendation(total_experience):
    if total_experience >= 5:
        return "Your performance and experience suggest that you're well-positioned for a salary increase. Consider discussing this with your manager during your next performance review."
    elif total_experience >= 3:
        return "You've gained valuable experience and have a solid performance rating. It might be a good time to explore opportunities for advancement within the company or discuss a salary review with your manager."
    else:
        return "Focus on enhancing your skills, gaining more experience, and improving your performance to increase your chances of a salary raise in the future."

# Route to render HTML form
@app.route('/')
def form():
    return render_template('form.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name']
        age = int(request.form['age'])
        gender = request.form['gender']
        designation = request.form['designation']
        unit = request.form['unit']
        past_experience = float(request.form['past_experience'])
        rating = float(request.form['rating'])
        date_of_join = request.form['date_of_join']

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

        preprocessed_data = preprocess_input(input_data)
        total_experience = preprocessed_data['TOTAL_EXP'].iloc[0]
        recommendation = generate_recommendation(total_experience)
        salary_prediction = model.predict(preprocessed_data)

        return render_template('result.html', name=name, prediction=salary_prediction[0], recommendation=recommendation)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
```

### HTML Templates
**form.html**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Salary Prediction App</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); width: 100%; max-width: 1000px; display: flex; justify-content: space-between; position: relative; }
        .form-section { flex: 1; padding: 20px; box-sizing: border-box; }
        .about-section { flex: 1; padding: 20px; box-sizing: border-box; text-align: justify; }
        .divider { position: absolute; left: 50%; top: 0; bottom: 0; width: 2px; background-color: #ccc; transform: translateX(-50%); }
        h1 { text-align: center; color: #333; margin-bottom: 20px; text-decoration: underline; }
        label { display: block; margin: 10px 0 5px; color: #333; }
        input[type="text"], input[type="number"], select, input[type="date"] { width: calc(100% - 22px); padding: 10px; margin-bottom: 10px; border: 1px solid #ccc; border-radius: 4px; }
        input[type="submit"] { width: 100%; padding: 10px; background-color: #5cb85c; border: none; border-radius: 4px; color: #fff; font-size: 16px; cursor: pointer; }
        input[type="submit"]:hover { background-color: #4cae4c; }
        .about-section h2 { color: #333; margin-bottom: 10px; }
        .about-section p { color

: #666; line-height: 1.6; }
    </style>
</head>
<body>
    <div class="container">
        <div class="form-section">
            <h1>Salary Prediction Form</h1>
            <form action="/predict" method="POST">
                <label for="name">Name:</label>
                <input type="text" id="name" name="name" required>
                <label for="age">Age:</label>
                <input type="number" id="age" name="age" required>
                <label for="gender">Gender:</label>
                <select id="gender" name="gender" required>
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                </select>
                <label for="designation">Designation:</label>
                <select id="designation" name="designation" required>
                    <option value="Analyst">Analyst</option>
                    <option value="Associate">Associate</option>
                    <option value="Senior Analyst">Senior Analyst</option>
                    <option value="Manager">Manager</option>
                    <option value="Senior Manager">Senior Manager</option>
                    <option value="Director">Director</option>
                </select>
                <label for="unit">Department:</label>
                <input type="text" id="unit" name="unit" required>
                <label for="past_experience">Past Experience (in years):</label>
                <input type="number" id="past_experience" name="past_experience" step="0.1" required>
                <label for="rating">Rating (0.0 to 5.0):</label>
                <input type="number" id="rating" name="rating" step="0.1" required>
                <label for="date_of_join">Date of Join:</label>
                <input type="date" id="date_of_join" name="date_of_join" required>
                <input type="submit" value="Predict Salary">
            </form>
        </div>
        <div class="divider"></div>
        <div class="about-section">
            <h2>About the Salary Prediction App</h2>
            <p>This application predicts the salary of an employee based on various factors such as age, gender, designation, department, past experience, and performance ratings. By utilizing a pre-trained machine learning model, it provides accurate salary predictions and recommendations for salary increases based on the total experience of the employee.</p>
        </div>
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
        body { font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); width: 100%; max-width: 600px; text-align: center; }
        h1 { color: #333; }
        p { color: #666; line-height: 1.6; }
        .recommendation { background-color: #e7f7e7; border: 1px solid #d4f1d4; border-radius: 4px; padding: 10px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Salary Prediction Result</h1>
        <p><strong>Name:</strong> {{ name }}</p>
        <p><strong>Predicted Salary:</strong> ${{ prediction }}</p>
        <div class="recommendation">
            <p><strong>Recommendation:</strong> {{ recommendation }}</p>
        </div>
    </div>
</body>
</html>
```

**error.html**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); width: 100%; max-width: 600px; text-align: center; }
        h1 { color: #e74c3c; }
        p { color: #666; line-height: 1.6; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Error</h1>
        <p>{{ error_message }}</p>
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
- Shreyas Gowda R: shreyasgowda9535@gmail.com

---

Feel free to adjust the README and code snippets as needed to better fit your project requirements.
