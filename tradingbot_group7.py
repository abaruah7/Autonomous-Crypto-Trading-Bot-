# -*- coding: utf-8 -*-
"""tradingbot_group7

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1szw7oTVkQ0VfP8UtF8q_h_oLJAMQ6KSJ

## Imports
"""

!pip3 install python-binance

# Commented out IPython magic to ensure Python compatibility.
import trading_helper as th
import numpy as np
import pandas as pd
import time
import os
# %tensorflow_version 2.x
import tensorflow
import tensorflow as tf
import json
from datetime import datetime, timedelta, timezone

def strategy(model, keys_fname, log_fname, win_sz, trading_interval='6h'):
    """
    Arguments
    --------- 
    model            : Some trained forecasting model
    keys_fname       : File name to the keys, see below
    log_fname        : You will need to log the trades you submit to Binance. 
                       The file name should be on the format "tradingbot_group<number>_<date_time>.log". 
                       E.g. "tradingbot_group1_2019-11-13_00:23:23.log"
    win_sz           : The window size that you specified when training the model, 
                       i.e. how many time steps of historical data the model needs for a forecast.
    trading_interval : How often the trading bot will query data from Binance, 
                       make a new forecast and decide whether to trade. 
                       E.g. '1min', '5min', '15min', '30min', '1h', '2h', '4h', '6h', '12h', '1d' or '1w'.
    """
    
    # use the keys stored in KEYS_FNAME to initialize the Binance client
    client = th.init_client(keys_fname)
    
    # convert the trading_interval string to seconds (int)
    interval_s = th.interval_string_to_seconds(trading_interval)
    
    # initialize the log file
    th.log(log_fname, new_log=True)
        
    # initialize trading with historical data
    t = datetime.utcnow()
    # timedelta(weeks=2) -> 2 weeks back in time
    t = th.UTC('time') - timedelta(weeks=2)
    # get data to begin forecasting from
    data = th.get_klines([t.year, t.month, t.day, t.hour, t.minute, t.second], interval=trading_interval)
    
    # forecast 
    x = data['returns'][-win_sz:].to_numpy().reshape((win_sz, 1))
    forecast = model.predict(x)
    
    while True:
 
        try:

            if forecast[-1] >= 0.05:  # forcasted big price increase, the average 6h window price increase from 2011 to 2019 is around 0.11%
                                      # there 243 instances that the 6h window price increase is greater than 5%, so we think 5% is a good cutoff 
                # decide on how to set the entry price
                entry_price = data['high'][-1] * 0.95    # since May 2019, the Bitcoin price has been dropping, if even the forecasted is positive
                                                         # we are not going to pay the full "high" price, but at a 5% discount price
                risk = 0.4  # modified risk
                # get the current balance
                balance = client.get_asset_balance(asset='USDT')
                entry_capital = np.float64(balance['free']) * risk # balance['free'] is a str -> convert to float
               
                if entry_capital > 0:
                    order = th.limit_buy(client, 'BTCUSDT', entry_price, quantity=entry_capital)
                   
                    th.log(log_fname, order_type='NEW_BUY', quantity=order['origQty'],
                        price=entry_price, time=th.UTC('iso'))
                  
            if 0.05 > forecast[-1] > 0.00075:  # forcasted price increase to cover the commission
                # decide on how to set the entry price
                entry_price = data['high'][-1] * 0.9      # purchase at a 10% discount
                risk = 0.3  # modified risk 
                # get the current balance
                balance = client.get_asset_balance(asset='USDT')
                entry_capital = np.float64(balance['free']) * risk # balance['free'] is a str -> convert to float

 

                if entry_capital > 0:
                     order = th.limit_buy(client, 'BTCUSDT', entry_price, quantity=entry_capital)
              
                     th.log(log_fname, order_type='NEW_BUY', quantity=order['origQty'],

                        price=entry_price, time=th.UTC('iso'))

            elif forecast[-1] < 0:  # forcasted price decrease
                exit_price = data['low'][-1] * 1.05       # we would like to sell at 5% premium
                balance = client.get_asset_balance(asset='BTC')
                risk = 0.5  # modified risk
                exit_capital = np.float64(balance['free']) * risk # balance['free'] is a str -> convert to float

                if entry_capital > 0:
                    order = th.limit_sell(client, 'BTCUSDT', exit_price, quantity=exit_capital)
                    th.log(log_fname, order_type='NEW_SELL', quantity=order['origQty'],
                        price=exit_price, time=th.UTC('iso'))
                   
            # pause/sleep for the trading_interval
            time.sleep(interval_s)
            # get last time from data
            t = data.index[-1]
            # update data 
            tmp = th.get_klines([t.year, t.month, t.day, t.hour, t.minute, t.second], interval=trading_interval)
            data.append(tmp)
            forecast = model.predict(data['returns'][-win_sz:])
                
        except ValueError:
            pass

# load pre-trainied model
model = tf.keras.models.load_model('BidirLSTM_2layer_regress_epochs75.h5')
keys_fname = 'group7_keys.json' # set this file name to your file name
log_fname = 'tradingbot_group7_%s.log' % (th.UTC('iso')[:-10]) # 
win_sz = 28  
trading_interval = '6h'
strategy(model, keys_fname, log_fname, win_sz=28, trading_interval=trading_interval)