# Libraries
import os
import torch
import random
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from itertools import product
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

# to install Pytorch on this PC 
# !pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Check what device is available
def check_device():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the name of the current device
        device_name = torch.cuda.get_device_name(0)
        print(f"Device Name: {device_name}. device = {device}")
    else:
        print(f"CUDA is not available. {device} is available")

    
    return device

# Set the seed for reproducibility
def set_the_seeds(seed = 1234):
    print(f'Setting the random, numpy and torch seed to {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



#######################################
##################LSTM#################
#######################################

# Class for LSTM network
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_sizes[2], 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x, _ = self.lstm3(x)
        x = self.fc(x[:, -1, :])
        return x
    
# Trains the LSTM for given dropout, lr and epochs
def train_model(X_train : pd.DataFrame, y_train : pd.DataFrame, DROPOUT_RATE = 0.2, LEARNING_RATE = 0.001, EPOCHS = 10):

    # Device used -> GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model and optimization criterion
    model = LSTMPredictor(input_size=X_train.shape[2], hidden_sizes=[128, 64, 32], dropout_rate = DROPOUT_RATE).to(device, non_blocking=True)
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    criterion = nn.MSELoss()

    # Pandas to Pytorch tensor
    X_train = torch.tensor(X_train, dtype = torch.float32).to(device, non_blocking=True)
    y_train = torch.tensor(y_train, dtype = torch.float32).to(device, non_blocking=True)
    
    # Train the model
    for epoch in range(EPOCHS):

        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train).squeeze()
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        #print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}")

    return model

# Performs single gridsearch iteration
def gridsearch_model(
    X_train : np.ndarray, y_train : np.ndarray, X_val : np.ndarray, y_val : np.ndarray,
    param_grid : dict,
    train_fn,        #user-provided train function
    predict_fn,      #user-provided predict function
    scaler_y : Optional[StandardScaler] = None,
    verbose : bool = True) -> Tuple[nn.Module, dict, float]: # best_model, best_params, best_score

    best_score = float('inf')
    best_model = None
    best_params = None

    param_keys, param_values = zip(*param_grid.items())

    for combo in product(*param_values):
        current_params = dict(zip(param_keys, combo))
        if verbose:
            print(f"Trying params: {current_params}")

        model = train_fn(X_train, y_train, X_val, y_val, current_params)
        y_val_pred = predict_fn(model, X_val)

        # Rescale if needed
        if scaler_y is not None:
            y_val_pred = scaler_y.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
            y_val_true = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
        else:
            y_val_true = y_val

        val_rmse = root_mean_squared_error(y_val_true, y_val_pred)

        if verbose:
            print(f"Validation RMSE: {val_rmse:.5f}")

        if val_rmse < best_score:
            best_score = val_rmse
            best_model = model
            best_params = current_params

    return best_model, best_params, best_score

def walk_forward_block_optimization(
    X: pd.DataFrame, y: pd.DataFrame,
    walk_forward_param_grid : dict = None,
    param_grid : dict = None,
    train_fn = None,
    predict_fn = None,
    verbose : bool = True)-> pd.DataFrame:

    assert train_fn is not None and predict_fn is not None, "You must provide training and prediction functions."

    # Parameters for walk forward block lengths and proportions
    n_blocks = walk_forward_param_grid['number_of_blocks']
    train_blocks = walk_forward_param_grid['train_blocks']
    val_blocks = walk_forward_param_grid['validation_blocks']
    test_blocks = walk_forward_param_grid['test_blocks']

    if train_blocks + val_blocks + test_blocks > n_blocks:
        raise ValueError('Incorrect number of blocks specified')

    n = len(X)
    block_size = n // n_blocks
    all_results = []

    for start_block in range(n_blocks - (train_blocks + val_blocks + test_blocks) + 1):
        train_start = start_block * block_size
        train_end = train_start + train_blocks * block_size
        val_end = train_end + val_blocks * block_size
        test_end = val_end + test_blocks * block_size

        X_train, y_train = X.iloc[train_start:train_end], y.iloc[train_start:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:test_end], y.iloc[val_end:test_end]

        if verbose:
            print(f"\nWalk {start_block+1}: Train[{train_start}:{train_end}], Val[{train_end}:{val_end}], Test[{val_end}:{test_end}]")

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled   = scaler_X.transform(X_val)
        X_test_scaled  = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()

        # Reshape if needed (e.g., for LSTM)
        if len(X_train_scaled.shape) == 2:
            X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
            X_val_scaled = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
            X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

        best_model, best_params, best_score = gridsearch_model(
            X_train_scaled, y_train_scaled,
            X_val_scaled, y_val_scaled,
            param_grid,
            train_fn=train_fn,
            predict_fn=predict_fn,
            scaler_y=scaler_y,
            verbose=verbose
        )

        # Predict on test
        y_test_pred = predict_fn(best_model, X_test_scaled)
        y_test_pred = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

        segment_results = pd.DataFrame({
            "timestamp": X_test.index,
            "true_value": y_test['target'],
            "predicted_value": y_test_pred
        })
        all_results.append(segment_results)

    final_results = pd.concat(all_results, ignore_index=True)
    final_results.index = final_results["timestamp"]
    return final_results[['true_value', 'predicted_value']]


# Example train_fn and predict_fn functions
def train_lstm_model(X_train, y_train, X_val, y_val, params):
    return train_model(
        X_train, y_train,
        DROPOUT_RATE=params.get("dropout_rate", 0.2),
        LEARNING_RATE=params.get("learning_rate", 0.001),
        EPOCHS=params.get("epochs", 10)
    )

def predict_lstm(model, X):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device, non_blocking=True)
    with torch.no_grad():
        y_pred = model(X_tensor).squeeze().cpu().numpy()
    return y_pred

