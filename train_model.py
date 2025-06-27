from data_preprocessing import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os
from collections import Counter

def train_and_save_model():
    print("🔍 Loading data...")
    X, y = load_data("ravdess-dataset")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📊 Class distribution before SMOTE: {Counter(y_train)}")

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SMOTE: Balance training data
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f"✅ After SMOTE: {Counter(y_train)}")

    # PCA: Reduce dimensionality
    pca = PCA(n_components=30)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Train XGBoost classifier
    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        eval_metric="mlogloss"
    )

    print("🚀 Training XGBoost with early stopping...")
    # clf.fit(
    #     X_train,
    #     y_train,
    #     eval_set=[(X_test, y_test)],
    #     early_stopping_rounds=10,
    #     verbose=True
    # )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Test Accuracy: {acc:.2f}")

    # Save model and preprocessors
    os.makedirs("saved_model", exist_ok=True)
    joblib.dump(clf, "saved_model/emotion_model.pkl")
    joblib.dump(scaler, "saved_model/scaler.pkl")
    joblib.dump(pca, "saved_model/pca.pkl")
    print("💾 Model and preprocessors saved in 'saved_model/'.")

if __name__ == "__main__":
    train_and_save_model()
