from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
app.secret_key = 'sdfdsfdsf3343223'  # Required for session management

# Load your model
model = pickle.load(open('model.pkl','rb'))
#model = pickle.load(open('stacking_classifier_model.pkl', 'rb'))

# Dummy admin credentials
ADMIN_USER = "admin"
ADMIN_PASS = "admin"

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USER and password == ADMIN_PASS:
            session['loggedin'] = True
            return redirect(url_for('input_form'))
        else:
            flash('Invalid Credentials. Please try again.')
    return render_template('login.html')

@app.route('/input', methods=['GET', 'POST'])
def input_form():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    prediction = None
    image_path = None
    if request.method == 'POST':
        try:
            # Collect form data
            form_data = {
                'age': [int(request.form['age'])],
                'number_of_sexual_partners': [int(request.form['number_of_sexual_partners'])],
                'first_sexual_intercourse': [int(request.form['first_sexual_intercourse'])],
                'number_of_pregnancies': [int(request.form['number_of_pregnancies'])],
                'smokes': [int(request.form['smokes'])],
                'smokes_years': [int(request.form.get('smokes_years', 0))],
                'smokes_packs_year': [int(request.form.get('smokes_packs_year', 0))],
                'hormonal_contraceptives': [int(request.form['hormonal_contraceptives'])],
                'hormonal_contraceptives_years': [int(request.form.get('hormonal_contraceptives_years', 0))],
                'iud': [int(request.form.get('iud', 0))],
                'iud_years': [int(request.form.get('iud_years', 0))],
                'stds': [int(request.form['stds'])],
                'hinselmann': [int(request.form.get('hinselmann', 0))],
                'schiller': [int(request.form.get('schiller', 0))],
                'citology': [int(request.form.get('citology', 0))],
            }
    
            # Convert to DataFrame with correct feature names
            input_df = pd.DataFrame(form_data)
    
            # Predict
            prediction = model.predict(input_df)[0]
            # Prediction probabilities
            prediction_probabilities = model.predict_proba(input_df)[0]
            # Coefficients
            coefficients = model.coef_[0]
    
            feature_names = list(form_data.keys())
            
            # Feature values for the input
            feature_values = input_df.iloc[0].values
            
            # Calculate contribution of each feature
            contributions = coefficients * feature_values
            
            df_contributions = pd.DataFrame({
                'Feature': feature_names,
                'Value': feature_values,
                'Coefficient': coefficients,
                'Contribution': contributions
            })
    
            # Sort by absolute contribution value
            df_contributions['Abs_Contribution'] = df_contributions['Contribution'].abs()
            df_contributions = df_contributions.sort_values(by='Abs_Contribution', ascending=False)
        
            # Create the plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Abs_Contribution', y='Feature', data=df_contributions, palette='viridis')
            plt.title('Factors Influencing This Prediction')
            plt.xlabel('Absolute Contribution Value')
            plt.ylabel('Feature')
            
            # Ensure static directory exists
            if not os.path.exists('static'):
                os.makedirs('static')
            
            # Save the plot as an image file
            image_path = 'static/feature_contribution.png'
            plt.savefig(image_path, bbox_inches='tight')
            
            # Close the plot to free up resources
            plt.close()
        except Exception as e:
            flash(f'Error processing input: {e}')

    return render_template('input_form.html', prediction=prediction, image_path=image_path)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
