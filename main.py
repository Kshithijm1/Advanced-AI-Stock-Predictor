from training import main_train
from prediction import main_predict, evaluate_performance
from data_fetching import fetch_stock_data, add_technical_indicators, preprocess_data

ticker = 'NVDA'

# Train the model (this should be done once, comment out after the first run)
main_train(ticker, fetch_stock_data, add_technical_indicators, preprocess_data)

# Predict the next week's prices
predictions = main_predict(ticker, fetch_stock_data, add_technical_indicators)
print(f"Predicted stock prices for the next week for {ticker}:")
for i, price in enumerate(predictions, 1):
    print(f"Day {i}: {price:.2f}")
