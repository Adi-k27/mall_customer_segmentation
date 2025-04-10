# src/model.py
import joblib
from sklearn.cluster import KMeans
from src.logger import get_logger
import os

logger = get_logger(__name__)

def train_kmeans(data, k):
    logger.info(f"Training KMeans with k={k}")
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(data)
    logger.info("KMeans training complete")
    return model, model.labels_, model.inertia_

def save_model(model, path="models/kmeans_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")

def load_model(path="models/kmeans_model.pkl"):
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model
