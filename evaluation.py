import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to evaluate a model's performance
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_true = y
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")
    
    plt.figure(figsize=(14,5))
    plt.plot(y_true, color='blue', label='Actual')
    plt.plot(y_pred, color='red', label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Prices')
    plt.show()

# Function to plot predictions against actual stock prices
def plot_predictions(stock_data, predictions, num_days=7):
    plt.figure(figsize=(14,5))
    plt.plot(stock_data['Close'][-60:], color='blue', label='Actual Stock Price')
    plt.plot(range(len(stock_data['Close'][-60:]), len(stock_data['Close'][-60:])+num_days), predictions, color='red', label='Predicted Stock Price')
    plt.legend()
    plt.title('Actual vs Predicted Stock Prices')
    plt.show()

# Function to evaluate the future performance of the model
def evaluate_future_performance(stock_data, predictions, num_days=7):
    y_true = stock_data['Close'][-num_days:].values
    mae = mean_absolute_error(y_true, predictions[:num_days])
    mse = mean_squared_error(y_true, predictions[:num_days])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, predictions[:num_days])
    
    print(f"Future Performance Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")
