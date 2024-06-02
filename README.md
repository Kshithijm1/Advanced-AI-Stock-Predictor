
# Advanced Stock Price Prediction Model

Welcome to the **Advanced Stock Price Prediction Model** repository! This project is designed to predict stock prices with high accuracy using a combination of machine learning and deep learning techniques. The model integrates technical indicators, tree-based models, and advanced neural networks to deliver state-of-the-art predictions.

## Features

- **Data Fetching and Preprocessing**: Automatically fetches historical stock data, computes essential technical indicators, and preprocesses the data for model training.
- **Advanced Model Architecture**: Utilizes a hybrid approach combining Convolutional Neural Networks (CNNs), Bidirectional Long Short-Term Memory (BiLSTM) networks, and Transformer layers for robust feature extraction and prediction.
- **Tree-based Model Integration**: Includes Random Forest, Gradient Boosting, and XGBoost models to enhance prediction accuracy.
- **Comprehensive Evaluation**: Offers detailed performance metrics and visualization tools for model evaluation and future performance assessment.

## Project Structure

- **`data_fetching.py`**: Handles data fetching from Yahoo Finance, computes technical indicators like RSI, Bollinger Bands, and more, and preprocesses data for model consumption.
- **`model_building.py`**: Contains functions to build and train tree-based models and the advanced CNN-BiLSTM-Transformer model. Also includes feature selection and model saving/loading utilities.
- **`training.py`**: The main script for training the models. Fetches and preprocesses data, trains the models, and saves the trained models and preprocessing artifacts.
- **`prediction.py`**: Facilitates predicting future stock prices using the trained models. It preprocesses input data, makes predictions, and inversely transforms the predictions to their original scale.
- **`evaluation.py`**: Provides functions for evaluating model performance using various metrics and visualizing actual vs. predicted prices.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model on a specific stock ticker, use the `main.py` script. This will fetch the data, train the models, and save the necessary artifacts.

```python
from training import main_train
from data_fetching import fetch_stock_data, add_technical_indicators, preprocess_data

ticker = 'NVDA'
main_train(ticker, fetch_stock_data, add_technical_indicators, preprocess_data)
```

### Predicting Stock Prices

After training the model, you can predict the next week's stock prices using the `main_predict` function.

```python
from prediction import main_predict
from data_fetching import fetch_stock_data, add_technical_indicators

ticker = 'NVDA'
predictions = main_predict(ticker, fetch_stock_data, add_technical_indicators)
print(f"Predicted stock prices for the next week for {ticker}:")
for i, price in enumerate(predictions, 1):
    print(f"Day {i}: {price:.2f}")
```

### Evaluating Model Performance

You can evaluate the model's performance on actual stock prices (if available) using the `evaluate_performance` function.

```python
from prediction import evaluate_performance

# Replace with actual stock prices for evaluation
actual_prices = [...]
mae, mse, rmse, r2 = evaluate_performance(predictions, actual_prices)
print(f"Future Performance Evaluation:
MAE: {mae:.2f}
MSE: {mse:.2f}
RMSE: {rmse:.2f}
R2: {r2:.2f}")
```

## Contribution

We welcome contributions to enhance the functionality and performance of this project. Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.

---
