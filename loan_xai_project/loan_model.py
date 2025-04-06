import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("data/train.csv")

# Drop ID
df.drop(columns=["Loan_ID"], inplace=True)

# Fill missing values
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
    df[col].fillna(df[col].mode()[0] if df[col].dtype == 'O' else df[col].median(), inplace=True)

# Convert '3+' to int
df['Dependents'].replace('3+', 3, inplace=True)
df['Dependents'] = df['Dependents'].astype(int)

# Encode categorical variables
cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# Train model
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "loan_model.pkl")
joblib.dump(encoder, "encoder.pkl")
