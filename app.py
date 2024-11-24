from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)