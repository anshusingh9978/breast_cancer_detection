import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # ✅ use joblib instead of pickle

# Load dataset
df = pd.read_csv('data/data.csv')

# Drop unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'])

# Encode target
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

# Select key features
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean',
    'area_mean', 'smoothness_mean', 'compactness_mean'
]
X = df[features]
y = df['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'RandomForest': RandomForestClassifier()
}

best_model = None
best_score = 0

print("Model Accuracies:\n")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    score = accuracy_score(y_test, preds)
    print(f"{name}: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_model = model

# ✅ Save best model and scaler using joblib
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("\nBest model saved successfully.")
