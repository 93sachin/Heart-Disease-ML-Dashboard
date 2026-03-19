import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

st.title("❤️ Heart Disease Prediction Dashboard")

# Load dataset
df = pd.read_csv("HeartDiseaseTrain-Test.csv")

df = df.drop_duplicates()
df = pd.get_dummies(df, drop_first=True)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),
    "Naive Bayes": Pipeline([
        ("model", GaussianNB())
    ]),
    "Decision Tree": Pipeline([
        ("model", DecisionTreeClassifier())
    ])
}

accuracy_results = {}

# 🔥 Confusion Matrices
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracy_results[name] = acc

    st.subheader(f"📊 Confusion Matrix - {name}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

# 🔥 Random Forest + ROC
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

accuracy_results["Random Forest"] = accuracy_score(y_test, rf_pred)

st.subheader("📊 Confusion Matrix - Random Forest")
cm = confusion_matrix(y_test, rf_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", ax=ax)
st.pyplot(fig)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, rf_prob)
auc = roc_auc_score(y_test, rf_prob)

st.subheader("📈 ROC Curve")
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
ax.plot([0,1], [0,1], linestyle="--")
ax.legend()
st.pyplot(fig)

# Accuracy Comparison
st.subheader("📊 Model Accuracy Comparison")
fig, ax = plt.subplots()
ax.bar(accuracy_results.keys(), accuracy_results.values())
plt.xticks(rotation=30)
st.pyplot(fig)