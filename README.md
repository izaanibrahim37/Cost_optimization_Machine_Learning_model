# Indian Cost Optimization - ML Model

## About

This project provides **personalized savings recommendations** for Indian households using a machine learning model trained on expense patterns. The **LightGBM regression model** predicts potential savings across various expense categories (Groceries, Transport, Eating Out, etc.) while accounting for **Indian purchasing power parity (PPP)** and **financial constraints**. **Bayesian optimization** ensures optimal hyperparameter tuning for accurate savings predictions.

---

## 📁 Folder Structure

```
Indian-Cost-Optimization/
├── data.csv                                # Dataset of Indian household financial records
├── indian_cost_optimization_data_mining.py # Main Python script
├── Indian_Cost_Optimization.ipynb          # Jupyter notebook version
└── README.md                               # Project documentation
```

---

## 🛠 Tech Stack

- **Core ML**: LightGBM (Gradient Boosting Framework)
- **Hyperparameter Tuning**: Bayesian Optimization
- **Explainable AI**: SHAP (SHapley Additive exPlanations)
- **Data Processing**: Pandas, NumPy
- **Evaluation**: scikit-learn (MAE, MSE, R², Precision/Recall)
- **Visualization**: Matplotlib, Seaborn

---

## ▶️ How to Run the Project

### 🐍 Python Script Version

```bash
pip install pandas numpy lightgbm bayesian-optimization shap matplotlib seaborn scikit-learn
python indian_cost_optimization_data_mining.py
```

### 📓 Jupyter Notebook Version

1. Upload `Indian_Cost_Optimization.ipynb` to Google Colab or Jupyter.
2. Run all cells sequentially.
3. View results including:
   - Model training progress
   - Savings recommendations
   - SHAP feature importance plots
   - Diagnostic visualizations

---

## 💡 Sample Input

```python
sample_input = {
    'Income': 1500000,          # Annual income in ₹
    'Age': 35,
    'Dependents': 2,
    'Occupation': 'Professional',
    'City_Tier': 'Tier_2',
    'Rent': 360000,             # Annual rent in ₹
    'Loan_Repayment': 180000,
    'Insurance': 60000,
    'Groceries': 240000,
    'Transport': 120000,
    'Eating_Out': 96000,
    'Entertainment': 60000,
    'Utilities': 72000,
    'Healthcare': 48000,
    'Education': 120000,
    'Miscellaneous': 60000,
    'Desired_Savings_Percentage': 0.25,
    'Desired_Savings': None
}
```

---

## 🚀 Key Features

- 🇮🇳 **PPP-adjusted expense modeling** for Indian context
- 💰 **Automatic rupee conversion** with category-specific adjustments
- 🔒 **Savings target validation** (capped at income level)
- 📊 **Visual diagnostics** with SHAP feature importance
- ✅ **Comprehensive model evaluation** metrics
- 🛡 **Safety checks** for realistic savings recommendations

> **Note:** Both script and notebook versions expect `data.csv` in the same directory when running.

---

Made with ❤️ for smart Indian financial planning!