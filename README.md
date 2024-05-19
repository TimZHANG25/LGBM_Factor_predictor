# Introduction

This project employs the LightGBM model to fit the monthly return data of the CSI 500 and the CSI All Market (000985.XSHG) stocks. The goal is to uncover the nonlinear relationship between various factors and stock returns. The selected factors fall into two main categories: momentum factors and fundamental factors. The dataset spans from January 2010 to February 2024.

# Datas

Facotr Data are based on CSI 500 stock market or CSI All Market (000985.XSHG)
X (Features): Factors obtained daily.  
Y (Label): Calculated based on the monthly returns

# Hyperparameter Tuning

Since the optimal settings for LightGBM's hyperparameters were unknown, we used variable lengths for the training set, validation set, and test set. A grid search was employed to determine the best values for the following key LightGBM hyperparameters:

max_depth
learning_rate
n_estimators
Prediction Logic

We predict the return ranking of the stock pool on a monthly/weekly basis. The monthly/weekly predicted AUC is calculated, and the average AUC is taken as the overall prediction performance.

# Backtesting Logic

For backtesting, each month/week we adjust the portfolio by selecting the top 10% and removing the last 10% of stocks based on the predictions.
