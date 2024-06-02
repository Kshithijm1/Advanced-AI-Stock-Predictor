from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Input, Flatten, concatenate, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Add
import joblib
import os
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
import tensorflow as tf

# Function for feature selection
def feature_selection(X, y):
    selector = SelectKBest(f_regression, k='all')  # Ensure all features are used
    X_new = selector.fit_transform(X, y)
    return X_new, selector

# Function to build tree-based models (RandomForest, GradientBoosting, XGBoost)
def build_tree_models(X, y):
    print("Building tree models...")

    X_new, selector = feature_selection(X, y)
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)  # Change n_estimators to tune model complexity
    gb_model = GradientBoostingRegressor(n_estimators=200, random_state=42)  # Change n_estimators to tune model complexity
    xgb_model = XGBRegressor(n_estimators=200, random_state=42)  # Change n_estimators to tune model complexity
    
    print("Training RandomForestRegressor...")
    rf_model.fit(X_new, y)
    print("RandomForestRegressor trained.")
    
    print("Training GradientBoostingRegressor...")
    gb_model.fit(X_new, y)
    print("GradientBoostingRegressor trained.")
    
    print("Training XGBRegressor...")
    xgb_model.fit(X_new, y)
    print("XGBRegressor trained.")
    
    print("Tree models built.")
    return rf_model, gb_model, xgb_model, selector

# Function to build a transformer encoder layer
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])
    
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

# Function to build an advanced model combining CNN, BiLSTM, and Transformer layers
def build_advanced_model(input_shape):
    print("Building advanced model with CNN, BiLSTM, and Transformer...")
    input_layer = Input(shape=input_shape)
    
    # Convolutional layers
    cnn_out = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
    cnn_out = Conv1D(filters=128, kernel_size=3, activation='relu')(cnn_out)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
    
    # Bidirectional LSTM layer
    lstm_out = Bidirectional(LSTM(units=128, return_sequences=True))(cnn_out)
    lstm_out = Dropout(0.3)(lstm_out)
    
    # Transformer encoder layer
    transformer_out = transformer_encoder(lstm_out, head_size=128, num_heads=4, ff_dim=128, dropout=0.3)
    transformer_out = Flatten()(transformer_out)
    
    # Dense layers
    dense_out = Dense(64, activation='relu')(transformer_out)
    dense_out = Dropout(0.3)(dense_out)
    
    # Output layer
    output_layer = Dense(1)(dense_out)
    
    # Compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    print("Advanced model built.")
    return model

# Function to save training data and models
def save_training_data(X, y, scaler, rf_model, gb_model, xgb_model, selector, ticker):
    print("Saving training data and models...")
    os.makedirs(ticker, exist_ok=True)
    np.save(os.path.join(ticker, f'{ticker}_X.npy'), X)
    np.save(os.path.join(ticker, f'{ticker}_y.npy'), y)
    joblib.dump(scaler, os.path.join(ticker, f'{ticker}_scaler.pkl'))
    joblib.dump(rf_model, os.path.join(ticker, f'{ticker}_rf_model.pkl'))
    joblib.dump(gb_model, os.path.join(ticker, f'{ticker}_gb_model.pkl'))
    joblib.dump(xgb_model, os.path.join(ticker, f'{ticker}_xgb_model.pkl'))
    joblib.dump(selector, os.path.join(ticker, f'{ticker}_selector.pkl'))
    print("Training data and models saved.")

# Function to save the advanced model
def save_model(model, ticker):
    print("Saving advanced model...")
    os.makedirs(ticker, exist_ok=True)
    model.save(os.path.join(ticker, f'{ticker}_advanced_model.h5'))
    print("Advanced model saved.")

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
