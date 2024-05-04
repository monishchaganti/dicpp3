import streamlit as st
import pandas as pd
import joblib

# Load the trained classification model
model = joblib.load('model_with_scaler.pkl')

# Define the Streamlit app
def main():
    st.title('Cardiovascular Disease Risk Prediction')

    # Display home image
    home_image = st.image('home.jpg', use_column_width=True)
    
    # Define layout with 70% for input and 30% for output
    col1, col2 = st.columns([7, 3])
    
    # Input section on the left
    with col1:
        st.sidebar.header('User Input')
        # Define input fields for user input
        age = st.sidebar.slider('Age', min_value=18, max_value=100, value=50)
        education = st.sidebar.slider('Education', min_value=1, max_value=4, value=2)
        sex = st.sidebar.radio('Sex', ['Male', 'Female'])
        cigsPerDay = st.sidebar.slider('Cigarettes per Day', min_value=0, max_value=70, value=10)
        BPMeds = st.sidebar.radio('BP Medication', ['No', 'Yes'])
        prevalentStroke = st.sidebar.radio('Prevalent Stroke', ['No', 'Yes'])
        prevalentHyp = st.sidebar.radio('Prevalent Hypertension', ['No', 'Yes'])
        diabetes = st.sidebar.radio('Diabetes', ['No', 'Yes'])
        totChol = st.sidebar.slider('Total Cholesterol', min_value=100, max_value=600, value=200)
        sysBP = st.sidebar.slider('Systolic Blood Pressure', min_value=80, max_value=300, value=120)
        diaBP = st.sidebar.slider('Diastolic Blood Pressure', min_value=40, max_value=150, value=80)
        BMI = st.sidebar.slider('BMI', min_value=10, max_value=50, value=25)
        heartRate = st.sidebar.slider('Heart Rate', min_value=40, max_value=150, value=75)
        glucose = st.sidebar.slider('Glucose', min_value=40, max_value=300, value=100)

        # Predict button
        if st.sidebar.button('Predict'):
            # Empty the home image
            home_image.empty()
            
            # Convert categorical variables to numerical
            sex_num = 1 if sex == 'Male' else 0
            BPMeds_num = 0 if BPMeds == 'No' else 1
            prevalentStroke_num = 0 if prevalentStroke == 'No' else 1
            prevalentHyp_num = 0 if prevalentHyp == 'No' else 1
            diabetes_num = 0 if diabetes == 'No' else 1

            # Store user input in a DataFrame
            user_input = pd.DataFrame({
                'age': [age],
                'education': [education],
                'sex': [sex_num],
                'cigsPerDay': [cigsPerDay],
                'BPMeds': [BPMeds_num],
                'prevalentStroke': [prevalentStroke_num],
                'prevalentHyp': [prevalentHyp_num],
                'diabetes': [diabetes_num],
                'totChol': [totChol],
                'sysBP': [sysBP],
                'diaBP': [diaBP],
                'BMI': [BMI],
                'heartRate': [heartRate],
                'glucose': [glucose]
            })

            # Perform prediction
            prediction = model.predict(user_input)

            # Display prediction result with increased size
            if prediction == 0:
                st.markdown("<h2 style='text-align: center; color: green;'>Low risk of cardiovascular disease</h2>", unsafe_allow_html=True)
                # Load and display an image for low risk with increased width
                st.image('low_risk_image.jpg', width=400)
            else:
                st.markdown("<h2 style='text-align: center; color: red;'>High risk of cardiovascular disease</h2>", unsafe_allow_html=True)
                # Load and display an image for high risk with increased width
                st.image('high_risk_image.jpg', width=400)

if __name__ == '__main__':
    main()
