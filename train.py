# train.py
import pandas as pd
from src.data_loader import load_data
from src.model import train_kmeans, save_model
from src.logger import get_logger

logger = get_logger("train")

def main():
    try:
        df = load_data("data/mall_customers.csv")
        df.drop(columns=["CustomerID"], inplace=True, errors="ignore")
        train_features = ["Age", "Annual_Income", "Spending_Score"]
        k = 5

        model, labels, inertia = train_kmeans(df[train_features], k)
        save_model(model)

    except Exception as e:
        logger.exception("Training failed")

if __name__ == "__main__":
    main()
