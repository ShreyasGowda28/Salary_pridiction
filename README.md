# Salary Prediction App

This project is a Flask web application that predicts the salary of an employee based on various input features such as age, gender, designation, department, past experience, and performance ratings. The model also provides recommendations based on the total experience of the employee.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
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

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request with your changes. Make sure to follow the existing code style and include appropriate tests.

## Contact
For any inquiries or feedback, please contact:
- Gokulnath UC: gokulnathuc@gmail.com
- R Bilwananda: bilwananda30@gmail.com
- Shalom Raj J: shalom.raj2747@gmail.com
- Shreyas Gowda S: shreyasgowda9535@gmail.com

---