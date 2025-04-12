import streamlit as st
import pickle
import numpy as np

# Load the trained model from the pickle file
model_file = pickle.load(open('logistic.pkl', 'rb'))
model = model_file['smote_lr']  

def predict(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status):
    # Convert all inputs to float type
    input_values = [gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]
    try:
        input_array = np.array([input_values]).astype(np.float64)
        prediction = model.predict_proba(input_array)
        pred = '{0:.{1}f}'.format(prediction[0][1], 2) 
        return float(pred)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

def main():
    st.title("Prediction")
    html_temp = """
    <div style="background-color:#e2062c ;padding:10px">
    <h2 style="color:white;text-align:center;">Stroke Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Collect input from user
    gender = st.text_input("gender [1 - male , 0 - female]")
    age = st.text_input("age")
    hypertension = st.text_input("hypertension [0 - No , 1 - Yes]")
    heart_disease = st.text_input("heart_disease [0 - No , 1 - Yes]")
    ever_married = st.text_input("ever_married [0 - No , 1 - Yes]")
    work_type = st.text_input("work_type [Govt_Job - 0, Never_Worked - 1, Private - 2, Self_Employed - 3,Children - 4]")
    residence_type = st.text_input("residence_type [Rural - 0, Urban - 1]")
    avg_glucose_level = st.text_input("avg_glucose_level")
    bmi = st.text_input("bmi")
    smoking_status = st.text_input("smoking_status [0 - No , 1 - Yes]")

    # HTML blocks for results
    safe_html = """
    <div style="background-color:#32CD32;padding:10px">
    <h2 style="color:white;text-align:center;">Low Stroke Risk (0)</h2>
    </div>
    """
    danger_html = """
    <div style="background-color:#DC3545;padding:10px">
    <h2 style="color:white;text-align:center;">High Stroke Risk (1)</h2>
    </div>
    """

    if st.button("Predict [0 - Low Risk, 1 - High Risk]"):
        output = predict(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status)
        if output is not None:
            if output > 0.5:
                st.markdown(danger_html, unsafe_allow_html=True)
            else:
                st.markdown(safe_html, unsafe_allow_html=True)
        else:
            st.warning("Please enter valid numeric input values.")

if __name__ == '__main__':
    main()
