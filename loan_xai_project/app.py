import gradio as gr
import joblib
import pandas as pd

model = joblib.load("loan_model.pkl")
encoder = joblib.load("encoder.pkl")

def predict_loan(gender, married, dependents, education, self_employed,
                 applicant_income, coapplicant_income, loan_amount,
                 loan_term, credit_history, property_area):
    
    input_dict = {
        'Gender': encoder.transform([gender])[0],
        'Married': encoder.transform([married])[0],
        'Dependents': int(dependents),
        'Education': encoder.transform([education])[0],
        'Self_Employed': encoder.transform([self_employed])[0],
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': encoder.transform([property_area])[0],
    }
    
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    return "Approved ✅" if prediction == 1 else "Not Approved ❌"

demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Dropdown(['Male', 'Female'], label="Gender"),
        gr.Dropdown(['Yes', 'No'], label="Married"),
        gr.Dropdown(['0', '1', '2', '3'], label="Dependents"),
        gr.Dropdown(['Graduate', 'Not Graduate'], label="Education"),
        gr.Dropdown(['Yes', 'No'], label="Self Employed"),
        gr.Number(label="Applicant Income"),
        gr.Number(label="Coapplicant Income"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Amount Term"),
        gr.Dropdown([1.0, 0.0], label="Credit History"),
        gr.Dropdown(['Urban', 'Rural', 'Semiurban'], label="Property Area"),
    ],
    outputs="text",
    title="Loan Approval Predictor with Explainable AI"
)

if __name__ == "__main__":
    demo.launch()
