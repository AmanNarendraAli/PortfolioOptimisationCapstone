import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
import pandas as pd

np.random.seed(123)
tf.random.set_seed(123)

# Model.py

class Model:
    def __init__(self):
        self.data = None
        self.model = None

    def __build_model(self, input_shape, outputs):
        model = Sequential([
            LSTM(64, input_shape=input_shape),
            Dense(outputs, activation='softmax')
        ])

        def sharpe_loss(y_true, y_pred):
            """
            Custom loss function to maximize Sharpe Ratio using excess returns.

            Parameters:
            - y_true: Concatenated tensor of (next_returns, next_rf)
            - y_pred: Predicted allocations

            Returns:
            - Negative Sharpe Ratio
            """
            # Assuming the last column in y_true is the risk-free rate
            actual_returns = y_true[:, :-1]
            rf = y_true[:, -1]

            # Compute portfolio returns
            portfolio_returns = tf.reduce_sum(y_pred * actual_returns, axis=1)

            # Compute excess returns
            excess_returns = portfolio_returns - rf

            # Compute Sharpe ratio
            mean_excess_return = K.mean(excess_returns)
            std_return = K.std(portfolio_returns)
            sharpe = mean_excess_return / (std_return + 1e-5)

            # Return negative Sharpe to minimize
            return -sharpe

        model.compile(loss=sharpe_loss, optimizer='adam')
        return model

    def train(self, data: pd.DataFrame):
        window_size = 50

        # Exclude 'Risk_Free_Rate_Daily' from assets
        data_assets = data.drop(columns=['Risk_Free_Rate_Daily'])

        # Calculate daily returns for assets only
        returns = data_assets.pct_change().fillna(0)
        # Combine asset prices, returns, and risk-free rates
        combined_data = pd.concat([data_assets, returns, data['Risk_Free_Rate_Daily']], axis=1)
        combined_data.columns = [f"{ticker}_price" for ticker in data_assets.columns] + \
                                [f"{ticker}_return" for ticker in data_assets.columns] + \
                                ['Risk_Free_Rate_Daily']

        sequences = []
        next_returns = []
        next_rf = []
        for i in range(len(combined_data) - window_size):
            seq = combined_data.iloc[i:i+window_size].values
            sequences.append(seq)
            # The next day's returns
            next_return = returns.iloc[i+window_size].values
            next_rf_rate = combined_data.iloc[i + window_size]['Risk_Free_Rate_Daily']
            next_returns.append(next_return)
            next_rf.append(next_rf_rate)
        sequences = np.array(sequences)
        next_returns = np.array(next_returns)
        next_rf = np.array(next_rf)

        if self.model is None:
            input_shape = (window_size, combined_data.shape[1])  # Includes Risk_Free_Rate_Daily
            num_assets = data_assets.shape[1]  # Number of assets excluding Risk_Free_Rate_Daily
            self.model = self.__build_model(input_shape, num_assets)

        # Concatenate returns and risk-free rate for y_true
        y_true = np.hstack((next_returns, next_rf.reshape(-1, 1)))  # Shape: (samples, num_assets + 1)

        # Train the model with early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        self.model.fit(sequences, y_true, epochs=100, batch_size=64, verbose=0, callbacks=[early_stop])

    def predict_allocation(self, input_sequence):
        # Ensure input_sequence has the correct shape (1, window_size, num_features)
        input_sequence = np.expand_dims(input_sequence, axis=0)
        allocation = self.model.predict(input_sequence)[0]
        # Ensure allocations sum to 1
        allocation = allocation / np.sum(allocation)
        return allocation
