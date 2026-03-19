import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

# Page config
st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

st.title("❤️ Heart Disease Prediction Dashboard")

# Load dataset
df = pd.read_csv("HeartDiseaseTrain-Test.csv")

# ------------------------------
# DATASET INFO
# ------------------------------
st.subheader("📊 Dataset Preview")
st.write(df.head())

st.subheader("📊 Target Distribution")
target_counts = df['target'].value_counts()
st.write(target_counts)
st.bar_chart(target_counts)

st.subheader("📈 Percentage Distribution")
percentage = df['target'].value_counts(normalize=True) * 100
st.write(percentage)

# ------------------------------
# SPLIT DATA
# ------------------------------
X = df.drop('target', axis=1)
y = df['target']

# Convert text to number
X = pd.get_dummies(X)

# Fill missing values
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# TRAIN MODELS
# ------------------------------
lr = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier()
nb = GaussianNB()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
y_pred_lr = lr.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_nb = nb.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)

# ------------------------------
# ACCURACY TABLE
# ------------------------------
st.subheader("📊 Model Accuracy Scores")

acc_lr = accuracy_score(y_test, y_pred_lr)
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_nb = accuracy_score(y_test, y_pred_nb)
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf = accuracy_score(y_test, y_pred_rf)

acc_df = pd.DataFrame({
    "Model": ["Logistic Regression", "KNN", "Naive Bayes", "Decision Tree", "Random Forest"],
    "Accuracy": [acc_lr, acc_knn, acc_nb, acc_dt, acc_rf]
})

st.write(acc_df)

# Best model
best_model = acc_df.loc[acc_df['Accuracy'].idxmax()]
st.success(f"🏆 Best Model: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.2f})")

# ------------------------------
# CONFUSION MATRICES
# ------------------------------
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

st.subheader("📊 Confusion Matrices")

plot_cm(y_test, y_pred_lr, "Logistic Regression")
plot_cm(y_test, y_pred_knn, "KNN")
plot_cm(y_test, y_pred_nb, "Naive Bayes")
plot_cm(y_test, y_pred_dt, "Decision Tree")
plot_cm(y_test, y_pred_rf, "Random Forest")

# ------------------------------
# ROC CURVE (Random Forest)
# ------------------------------
st.subheader("📈 ROC Curve (Random Forest)")

y_prob = rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0,1], [0,1], linestyle='--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()

st.pyplot(fig)

# ------------------------------
# ACCURACY BAR CHART
# ------------------------------
st.subheader("📊 Model Accuracy Comparison")

fig, ax = plt.subplots()
sns.barplot(x="Model", y="Accuracy", data=acc_df, ax=ax)
plt.xticks(rotation=30)
st.pyplot(fig)