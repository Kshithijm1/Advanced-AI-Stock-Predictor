import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import tensorflow as tf
import os

# Function to predict next week's prices using the trained models
def predict_next_week(model, rf_model, gb_model, xgb_model, X, scaler, selector):
    print("Predicting next week's prices...")
    predictions = []

    current_batch = X[-1].reshape((1, X.shape[1], X.shape[2]))
    
    for i in range(7):
        # Flatten the features for tree models
        X_flat = current_batch.reshape((current_batch.shape[0], -1))
        selected_features = selector.transform(X_flat)
        
        # Predict with tree models
        rf_pred = rf_model.predict(selected_features).reshape(-1, 1)
        gb_pred = gb_model.predict(selected_features).reshape(-1, 1)
        xgb_pred = xgb_model.predict(selected_features).reshape(-1, 1)
        
        # Ensure the tree predictions are of the correct shape
        rf_pred_expanded = np.repeat(rf_pred, X.shape[1], axis=1).reshape((1, X.shape[1], 1))
        gb_pred_expanded = np.repeat(gb_pred, X.shape[1], axis=1).reshape((1, X.shape[1], 1))
        xgb_pred_expanded = np.repeat(xgb_pred, X.shape[1], axis=1).reshape((1, X.shape[1], 1))
        
        # Combine features with predictions
        combined_features = np.concatenate((current_batch, 
                                            rf_pred_expanded, 
                                            gb_pred_expanded, 
                                            xgb_pred_expanded), axis=2)
        
        # Predict with LSTM model
        lstm_pred = model.predict(combined_features)[0]
        predictions.append(lstm_pred[0])
        
        # Create the new feature vector with the LSTM prediction appended
        new_feature = np.append(combined_features[0, -1, :-3], lstm_pred).reshape(1, 1, -1)
        
        # Ensure the new feature vector matches the shape of current_batch along the feature axis
        if new_feature.shape[2] != current_batch.shape[2]:
            new_feature = new_feature[:, :, :current_batch.shape[2]]

        # Update the current batch to include the new prediction
        next_input = np.concatenate((current_batch[:, 1:, :], new_feature), axis=1)
        current_batch = next_input
    
    # Create a dummy array for inverse transform
    dummy_data = np.zeros((len(predictions), scaler.n_features_in_))
    dummy_data[:, 0] = predictions
    dummy_data = scaler.inverse_transform(dummy_data)
    predictions = dummy_data[:, 0]
    
    return predictions

# Main function to load the model, fetch stock data, preprocess, and predict
def main_predict(ticker, fetch_stock_data, add_technical_indicators):
    print(f"Loading model and data for {ticker}...")
    model, scaler, rf_model, gb_model, xgb_model, selector = load_trained_model(ticker)
    
    print("Fetching stock data for prediction...")
    stock_data = fetch_stock_data(ticker, period='90d')
    stock_data = add_technical_indicators(stock_data)
    
    print("Preprocessing data for prediction...")
    data = stock_data.values
    # Ensure that the data has the same number of features as used during training
    if data.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Expected {scaler.n_features_in_} features, but got {data.shape[1]} features")
    
    scaled_data = scaler.transform(data)
    
    X = []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, :])
    X = np.array(X)
    
    if len(X) == 0:
        raise ValueError("Not enough data to generate input sequences. Ensure you have at least 60 data points.")
    
    print("Data preprocessed for prediction.")
    predictions = predict_next_week(model, rf_model, gb_model, xgb_model, X, scaler, selector)
    
    print("Next week's prices predicted.")
    return predictions

# Function to evaluate the performance of predictions
def evaluate_performance(predictions, actual_prices):
    mae = mean_absolute_error(actual_prices, predictions)
    mse = mean_squared_error(actual_prices, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_prices, predictions)
    return mae, mse, rmse, r2

# Function to load the trained model and scaler
def load_trained_model(ticker):
    print("Loading trained model and scaler...")
    scaler = joblib.load(os.path.join(f'{ticker}', f'{ticker}_scaler.pkl'))
    rf_model = joblib.load(os.path.join(f'{ticker}', f'{ticker}_rf_model.pkl'))
    gb_model = joblib.load(os.path.join(f'{ticker}', f'{ticker}_gb_model.pkl'))
    xgb_model = joblib.load(os.path.join(f'{ticker}', f'{ticker}_xgb_model.pkl'))
    selector = joblib.load(os.path.join(f'{ticker}', f'{ticker}_selector.pkl'))
    model = tf.keras.models.load_model(os.path.join(f'{ticker}', f'{ticker}_advanced_model.h5'))
    print("Trained model and scaler loaded.")
    return model, scaler, rf_model, gb_model, xgb_model, selector
