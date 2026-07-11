import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv("data/loan_data.csv")

print("Head of the table")
print(df.head())
print("Columns of the table")
print(df.columns)
print("Shape of the table")
print(df.shape)
print("Data type of the table")
print(df.dtypes)
print("Missing Values of the table")
print(df.isnull().sum())
print("Information about the table")
print(df.info())
print("Loan status values count")
print(df["Loan_Status"].value_counts())

print("\nUnique Values:")
for column in df.columns:
    print(f"{column}: {df[column].unique()}")

df.drop("Loan_ID", axis=1, inplace=True)
print(df.columns)

df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)

# Numerical columns
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

print(df.isnull().sum())

df["Gender"] = df["Gender"].map({"Male": 1,"Female": 0})
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1,"N": 0})
df["Married"] = df["Married"].map({"Yes": 1,"No": 0})
df["Education"] = df["Education"].map({"Graduate": 1,"Not Graduate": 0})
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1,"No": 0})

le = LabelEncoder()
df["Property_Area"] = le.fit_transform(df["Property_Area"])

print(df.head())
print(df.dtypes)
print(df["Dependents"].unique())
df["Dependents"] = df["Dependents"].replace("3+", "3")
df["Dependents"] = df["Dependents"].astype(int)
print(df.dtypes)

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

print(X.head())

print()

print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

model = LogisticRegression(max_iter=4000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predicted:")
print(y_pred[:10])

print()

print("Actual:")
print(y_test.values[:10])

print(model.score(X_test, y_test))

print("Accuracy" , accuracy_score(y_test, y_pred))

print("Confusion Matrix \n " , confusion_matrix(y_test, y_pred))

print("Classification Report \n " , classification_report(y_test, y_pred))

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)
print(dt_model.score(X_test, y_test))
print(dt_model.predict(X_test))
dt_y_pred = dt_model.predict(X_test)

print("Decision Tree Accuracy:")
print(accuracy_score(y_test, dt_y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, dt_y_pred))

print("\nClassification Report")
print(classification_report(y_test, dt_y_pred))

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

print("Random Forest Accuracy:")
print(accuracy_score(y_test, rf_y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_y_pred))

print("\nClassification Report")
print(classification_report(y_test, rf_y_pred))

lr_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Logistic Regression Scores:", lr_scores)
print("Average Accuracy:", lr_scores.mean())

dt_scores = cross_val_score(
    dt_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Decision Tree Scores:", dt_scores)
print("Average Accuracy:", dt_scores.mean())

rf_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Random Forest Scores:", rf_scores)
print("Average Accuracy:", rf_scores.mean())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
scaled_model = LogisticRegression(max_iter=1000)

scaled_model.fit(X_train_scaled, y_train)

scaled_pred = scaled_model.predict(X_test_scaled)

print("Scaled Logistic Regression Accuracy:")
print(accuracy_score(y_test, scaled_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, scaled_pred))

print("\nClassification Report")
print(classification_report(y_test, scaled_pred))

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("Before SMOTE")
print(y_train.value_counts())

print()

print("After SMOTE")
print(y_train_smote.value_counts())

lr_smote = LogisticRegression(max_iter=1000)

lr_smote.fit(X_train_smote, y_train_smote)

lr_pred = lr_smote.predict(X_test)

dt_smote = DecisionTreeClassifier(random_state=42)

dt_smote.fit(X_train_smote, y_train_smote)

dt_pred = dt_smote.predict(X_test)

rf_smote = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

rf_smote.fit(X_train_smote, y_train_smote)

rf_pred = rf_smote.predict(X_test)
print("Logistic Regression with SMOTE Accuracy:")
print(accuracy_score(y_test, lr_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, lr_pred))

print("\nClassification Report")
print(classification_report(y_test, lr_pred))

print("Decision Tree with SMOTE Accuracy:")
print(accuracy_score(y_test, dt_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, dt_pred))

print("\nClassification Report")
print(classification_report(y_test, dt_pred))

print("Random Forest with SMOTE Accuracy:")
print(accuracy_score(y_test, rf_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

print("Before SMOTE")
print(y_train.value_counts())

print("\nAfter SMOTE")
print(y_train_smote.value_counts())

joblib.dump(rf_smote, "models/loan_approval_model.pkl")
print("Model Saved Successfully!")
joblib.dump(scaler, "models/scaler.pkl")
