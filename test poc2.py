import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df["Age"] = pd.to_datetime("today").year - pd.to_datetime(df["DOB"], errors="coerce").dt.year
    df["Tenure (Years)"] = pd.to_datetime("today").year - pd.to_datetime(df["Joining Date"], errors="coerce").dt.year
    df.drop(columns=["Employee UID (masked)", "Resignation Date", "LWD", "Last Promotion Date"], errors="ignore", inplace=True)
    df.fillna(df.median(), inplace=True)
    return pd.get_dummies(df, drop_first=True)

df = pd.read_csv("Test data Prediction model.csv")
df = preprocess_data(df)
X = df.drop(columns=["Status"], errors="ignore")
y = df["Status"].apply(lambda x: 1 if x != "Active" else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(model, "logistic_regression_model.pkl")

def predict_attrition(employee_data):
    df_input = pd.DataFrame([employee_data])
    df_input = preprocess_data(df_input).reindex(columns=X.columns, fill_value=0)
    probability = model.predict_proba(scaler.transform(df_input))[:, 1][0]
    return probability * 100
