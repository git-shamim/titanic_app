import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("data/train.csv")

# Select relevant features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
df = df[["Survived"] + features].dropna()

# ✅ Encode categorical variables with explicit mappings
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# ✅ Add child flag (children under 12)
df["is_child"] = df["Age"].apply(lambda x: 1 if x < 12 else 0)
features.append("is_child")

# ✅ Optional: scale continuous features (tree models don’t require it, but good for completeness)
# scaler = StandardScaler()
# df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

# Define X and y
X = df[features]
y = df["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest + Grid Search
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "min_samples_split": [2, 5],
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

# Save best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, "titanic_model.pkl")
print("✅ Model trained and saved as titanic_model.pkl")
