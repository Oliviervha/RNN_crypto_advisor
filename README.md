# RNN_crypto_advisor
Using a recurrent neural network (RNN) to predict whether a crypto currency will increase on next trading day based on historical price and volume data.
* RNN_classifier.py - The main script where the model is trained and evaluation results are generated. 
* config.py - Here are all configuration settings related to the data, model and strategy located.  

Currently a pretrained model is running on a server and daily advice on whether to sell, buy or hold can be received by subscribing on [this](https://ai-crypto-advisor.webflow.io/) webpage.

# Strategy
The strategy is simple. Each day the model predicts whether the crypto price of the next day will increase with a certain probability. Based on a lower bound that can be set in the configuration file the model will buy crypto. 

# Data
The training dataset contains historical data of the daily closing price and volume of the Bitcoin (BTC) and one other coin named VeThor Token (VTHO). As it was discovered that the the alternative coin follows the Bitcoin trend we tried to predict VeThor Token by using historical Bitcoin data. 

# Model
A Recurrent Neural Network is used to train a model based on historical price and volume data. In the current configuration settings we chopped the data in pieces where for each day in the dataset the closing price of 5 consecutive days is used to predict the 6th day.  

# Evaluation
Evaluation is done based on the return by actively following the models advice. Here we look at the trend, which is the percentage difference between the first and the last closing price from the test set. The investment and reinvestment return is the return we will get following the trading strategy of our model. Then the random expected return is added to see if we perform better than random trading. 

* Trend: 1.27%
* Investment return: 10.57%
* Reinvestment return: 7.26%
* Random exp. return: 4.19%

![alt text](https://github.com/Oliviervha/RNN_crypto_advisor/blob/main/plot.png?raw=true)

![alt text](https://github.com/Oliviervha/RNN_crypto_advisor/blob/main/plot_rand.png?raw=true)
