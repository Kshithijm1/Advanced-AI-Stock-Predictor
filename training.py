from data_fetching import fetch_stock_data, add_technical_indicators, preprocess_data
from model_building import build_tree_models, build_advanced_model, save_training_data, save_model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Main training function
def main_train(ticker, fetch_stock_data, add_technical_indicators, preprocess_data):
    print(f"Starting training for {ticker}...")
    
    # Fetch and preprocess data
    stock_data = fetch_stock_data(ticker, period='2y')
    stock_data = add_technical_indicators(stock_data)
    X, y, scaler = preprocess_data(stock_data)
    
    # Build tree models
    X_flat = X.reshape((X.shape[0], -1))
    rf_model, gb_model, xgb_model, selector = build_tree_models(X_flat, y)
    
    # Prepare data for the advanced model
    rf_pred = rf_model.predict(selector.transform(X_flat)).reshape(-1, 1)
    gb_pred = gb_model.predict(selector.transform(X_flat)).reshape(-1, 1)
    xgb_pred = xgb_model.predict(selector.transform(X_flat)).reshape(-1, 1)
    
    rf_pred = np.repeat(rf_pred, X.shape[1], axis=1).reshape((X.shape[0], X.shape[1], 1))
    gb_pred = np.repeat(gb_pred, X.shape[1], axis=1).reshape((X.shape[0], X.shape[1], 1))
    xgb_pred = np.repeat(xgb_pred, X.shape[1], axis=1).reshape((X.shape[0], X.shape[1], 1))

    X_combined = np.concatenate((X, rf_pred, gb_pred, xgb_pred), axis=2)
    
    # Build advanced model
    model = build_advanced_model((X_combined.shape[1], X_combined.shape[2]))
    
    # Train advanced model
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(X_combined, y, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stop])
    
    # Save models and data
    save_training_data(X, y, scaler, rf_model, gb_model, xgb_model, selector, ticker)
    save_model(model, ticker)
    print(f"Training complete for {ticker}. Model and training data saved.")
