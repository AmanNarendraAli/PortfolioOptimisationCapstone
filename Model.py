import numpy as np

# setting the seed allows for reproducible results
np.random.seed(123)

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

import pandas as pd

class Model:
    def __init__(self):
        self.data = None
        self.model = None
    
    def __build_model(self, input_shape, outputs):
        '''
        Builds and returns the Deep Neural Network that will compute the allocation ratios
        that optimize the Sharpe Ratio of the portfolio
        
        inputs: input_shape - tuple of the input shape, outputs - the number of assets
        returns: a Deep Neural Network model
        '''
        model = Sequential([
        LSTM(64, input_shape=input_shape),
        Flatten(),
        Dense(outputs, activation='softmax')
    ])
        def sharpe_loss(_, y_pred):
            # Normalize time-series (make all time-series start at 1)
            data = tf.divide(self.data, self.data[0])  # data[0] is the first price point (normalized)
            
            # Value of the portfolio after allocations are applied
            portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1)
            
            # Calculate portfolio returns (avoid dividing by zero)
            portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / tf.maximum(portfolio_values[:-1], 1e-7)
            
            # Calculate Sharpe ratio (mean returns / standard deviation, avoid division by zero)
            sharpe = K.mean(portfolio_returns) / (K.std(portfolio_returns) + 1e-7)
            
            # Negate because we want to maximize Sharpe (minimizing the negative)
            return -sharpe

        
        model.compile(loss=sharpe_loss, optimizer='adam')
        return model
    
    def get_allocations(self, data: pd.DataFrame):
        lookback_days = 50
        # Extract 50-day window of prices and returns
        data_w_ret = np.concatenate([data.iloc[-lookback_days:].values, data.pct_change().iloc[-lookback_days:].values], axis=1)
        self.data = tf.cast(tf.constant(data.iloc[-lookback_days:]), float)

        if self.model is None:
            self.model = self.__build_model(data_w_ret.shape, len(data.columns))

        fit_predict_data = data_w_ret[np.newaxis, :]      
        self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=100, shuffle=False)
        return self.model.predict(fit_predict_data)[0]
