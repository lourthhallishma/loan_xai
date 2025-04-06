# Explainable AI for Loan Approval Transparency

This project is an end-to-end **Explainable AI system** designed to predict **loan approval** decisions and make them **transparent and interpretable** for users using SHAP and a Gradio-powered UI.

---

## Project Highlights

- **Data Cleaning & Feature Engineering**  
  Categorical variables like `Gender`, `Married`, and `Property_Area` are encoded and missing values are imputed logically.

- **Model Training with Random Forest**  
  A robust and interpretable Random Forest model is trained to predict loan approvals.

- **Explainability with SHAP (SHapley Additive exPlanations)**  
  SHAP is used to interpret the impact of each feature on individual loan decisions.

- **User Interface with Gradio**  
  A clean and interactive UI allows users to enter loan application details and instantly see predictions.

---

## What Does It Do?

- Predicts whether a loan will be **Approved** or **Not Approved**
- Explains **why** a decision was made using feature importance (with SHAP)
- Provides insights into which features most affect loan outcomes

---

## Key Findings from the Dataset

**Credit History** is the most influential factor — applicants with good history are highly likely to be approved.
**Applicant Income** does not always guarantee approval; creditworthiness matters more.
**Property Area** and **Loan Amount** also contribute but with less impact than credit history.

---
