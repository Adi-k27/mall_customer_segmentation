# 🏦 Loan Approval Prediction

This project uses Logistic Regression and Random Forest to predict whether a loan should be approved based on applicant data.

## ✅ Requirements
Install dependencies with:
```
pip install -r requirements.txt
```

## 💠 How to Use

1. Place your dataset as `loan_data.csv` in the `data/` folder.

2. Train the models:
   ```
   python train.py
   ```

3. Launch the Streamlit app:
   ```
   streamlit run app.py
   ```

## 📂 Folder Structure

- `data/`: Contains your dataset.
- `models/`: Trained models are saved here.
- `logs/`: Logging the activities and error.
- `src/`: All modular code (preprocessing, training, evaluation, logging).
- `app.py`: Streamlit app entry point.
- `train.py`: Model training script.
