import joblib

def load_model():
    model = joblib.load("saved_model/emotion_model.pkl")
    scaler = joblib.load("saved_model/scaler.pkl")
    pca = joblib.load("saved_model/pca.pkl")
    return model, scaler, pca
