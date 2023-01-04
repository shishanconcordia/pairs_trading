import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import numpy as np
import pandas as pd
# import pyfolio as pf
import statsmodels
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts 
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import json


def find_pairs(dt, p_v = 0.001, name = 'nasdaq_stock_pairs.json'):
    rst = {}
    length = dt.shape[1]
    for i in range(0, length - 1):
        temp = []
        for j in range(i+1, length - 1):
            try:
                X = dt.iloc[:, j]
                X = sm.add_constant(X)
                model = sm.OLS(dt.iloc[:, i], X)
                lm_rst = model.fit()
                result = adfuller(lm_rst.resid.values)
                if result[1] < p_v:
                    temp.append((dt.columns[j], result[1]))
            except:
                continue
        if temp:
            rst[dt.columns[i]] = temp
        
    with open(name, 'w') as f:
        json.dump(rst, f)
    return rst


def test_two_pairs(x_1, x_2):

    X = x_1
    X = sm.add_constant(X)
    model = sm.OLS(x_2, X)
    lm_rst = model.fit()
    result = adfuller(lm_rst.resid.values, regression = 'n')
    return result[1], lm_rst, result[0]


def show_pairs(x1, x2, start= '2022-06-01', end= '2022-12-01', intv = '1h', plotting = False):
    raw = yf.download([x1, x2], interval= intv, start=start, show_errors=False).dropna()

    dt = raw.loc[start:end,:]
    # print(dt.tail(3))
    test_dt = raw.loc[end:,:]
    X = np.log(dt.loc[:,'Adj Close'].iloc[:, 0])
    X = sm.add_constant(X)
    # X = dt.Close.iloc[:, 1]
    model = sm.OLS(np.log(dt.loc[:,'Adj Close'].iloc[:, 1]), X)
    rst = model.fit()
    # print(rst.summary())
    params = rst.params

    test_X = np.log(test_dt.loc[:,'Adj Close'].iloc[:, 0])
    test_X = sm.add_constant(test_X)
    test_rst = np.log(test_dt.loc[:,'Adj Close'].iloc[:, 1])- rst.predict(test_X)
    if plotting:
        print(dt.tail(3))
        temp_dt = (raw.loc[:,'Adj Close'])#/raw.loc[:,'Adj Close'].iloc[0])
        (temp_dt.iloc[:, 0] - temp_dt.iloc[:, 1]).plot(figsize = (19, 7))
        # (dt.Close).plot(figsize = (19, 12))
        plt.show()
        print(rst.summary())
        return rst, dt, test_X, test_rst
    else:
        return rst
    
def adf_rst(rst):
    result = adfuller(rst)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    return result[1]

