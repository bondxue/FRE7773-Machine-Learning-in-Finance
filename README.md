# FRE7773-Machine-Learning-in-Finance
machine learning homeworks 

---------------------------

### homework 1: daily stock return analysis using CAPM model

#### Dataset
`stock-treasury-2004_2006.csv`- For each trading day in the study period the dateset contains:  
   + `TREAS_3M`: the yield of the 3-month treasury note in percent (i.e 2.1 means 2.1%)
   + `Adjusted close price` of ten major stocks: GM, F, UTX, CAT, MRK, PFE, IBM, MSFT, C, XOM
   + `SP`: The S&P 500 equity index level  

#### Model  
`Capital Asset Pricing Model (CAPM)`
   + Calculate excess return for each stocks and SP index.
   + Conduct **linear regression** using `Scikit-Learn` and `Statsmodels` for each stock, using its excess return as the *y-variable* and the SP index excess return as the *x-variable*. Linear regression results using `Statsmodels` are shown as follows:

   <img src="https://github.com/bondxue/FRE7773-Machine-Learning-in-Finance/blob/master/HW01-LinearRegression/images/LinearReg.PNG" width="500">
   
   + Test whether `CAPM` is a good model for explaning daily stock return variance.
    

    
#### Results
* **Compare the two data frames and assert that they return the same values for $\alpha$ and $\beta$.**
  + Yes, the two data frames return the same values for $\alpha$ and $\beta$.
* **Based on the p-values, can we reject the null hypotheses alpha=0 or beta=0?**
  + The most common significance level to reject the null hypothesis is 0.05. We can see that for  $\alpha$, p-values are all greater than 0.05. Thus we hold the null hypotheses  that $\alpha = 0$. For $\beta$, p-values are all less than 0.05, thus we reject the null hypotheses that $\beta = 0$.
* **Is CAPM a good model for explaning daily stock return variance?**
  +  CAPM uses a $\beta$ to compensate investors for the risk they take. A high $\beta$ means that the asset is greatly affected by macro-economic changes, so the variance will be high. A low $\beta$ means that the asset is not heavily affected by market changes, so the expected return can also be lower. However, it is a too simple model that only uses one $\beta$ to explain the stock return. To fully explain stock return variance, I think more $\beta's$ need to be added into the model. 
  + We could check R-squared values and find that all are less than 0.5. It shows that CAPM model is not good fit for our daily stock return data.
    
---------------------------

### homework 2: wine quality analysis using OLS, bayes, and SGD

#### Dataset
`winequality-white.csv` -It contains 4898 observations of 11 numerical white wine attributes. It also contains a quality score that ranges from 0 to 10.

#### Models
compare three models, **OLS**, **Bayes** and **SGD** in predicting quality in terms of the other 11 features.

#### Precedure
1. Normalize all input features and the output variable quality using the `Scikit-Learn MinMaxScaler`
2. Display a description of the dataset
3. Split the dataset, keeping 1000 observations for out-of-sample testing
4. Fit **OLS**, **Bayes** and **SGD** on the training sample and display the weights
<img src="https://github.com/bondxue/FRE7773-Machine-Learning-in-Finance/blob/master/HW02-BayesianSGDRegression/images/weights.png" width="700">
5. For each model compute and print out the `RMSE` and `EVAR (explained variance score)` in (train) and out of (test) sample
6. Plot the `learning curves` for each model
<img src="https://github.com/bondxue/FRE7773-Machine-Learning-in-Finance/blob/master/HW02-BayesianSGDRegression/images/learning-curve.png" width="700">
7. Based on the evidence (`learning curves`, `RMSE`, `EVAR`) discuss which model is the best

#### Results
Based on learning curves, RMSE and EVAR, I think **SGD** is the best model in this case. 
1. From `RMSE` and `EVAR` we can see that **OLS** and **Bayes** perform similarly, and **SGD** obtains relatively higher `RMSE` and relatively lower `EVAR` scores in both training and test sets, which means **OLS** and **Bayes** are more accurate models in predicting the quality of wines.  
2. From learning curves, all of the three models perform well. We could see that the gaps between the `learning curve` of training and validation sets are all very narrow, also the overall error level is low, which means that all three models can be well generalized. However, we could see that for **OLS** and **Bayes**, the `learning cruves` converge to a little bit lower error level, which shows these two models may still outperform in terms of `learning curves`.  

---------------------------

### homework 3: Pima Indians Diabetes forecast using linear-SVM and rbf-SVM

#### Dataset
`pima-indians-diabetes.csv` -  forecast the occurence of diabetes from eight numerical features. Metadata for the features are found in the `README.txt` file.

#### Models
**Linear SVM** and **Gaussian RBF SVM model**

#### Precedure
1. Load and describe the data from file "Data/PimaIndiansDiabetes/pima-indians-diabetes.csv".
2. Split the data and retain 25% for testing
3. Fit a Linear SVM and and a Gaussian RBF SVM model.
4. For each model output the *in- and out-of-sample confusion matrix, accuracy, precision, recall and F1-score*
    
#### Results
In the showing case, we could see that **linear SVM** is better performed than **rbf SVM** since the out sample *F1* of **linear SVM** is slightly higher than **rbf SVM**. I tried to change $random\_state$ parameter in $train\_test\_split$ and test results for different training test splits. My expectation is **rbf SVM** will perform better. However, I find that:
+ For in sample set, **rbf SVM** will always perform better than **linear SVM**, since **rbf** could provide *nonlinear* classification which is more powerful. 
+ For out sample set, the performance of two SVMs will vary based on the different train-test splits.  Sometimes **rbf** is better, while sometimes **linear** is better. 
+ I think the reason is **rbf** is susceptible to *overfitting/training* issues  and also the dataset size is too small and therefore, the training set difference will affect the model performance. Sometimes, poor training set will cause **rbf** overfitting and having large bais. Thus, **rbf SVM** may perform poorer than **linear SVM** on out sample set. 
+ Therefore, we may need larger dataset or tuning hyperparameters for **rbf SVM** to make it more generalized, accordingly, its performance will be improved. 


---------------------------

### homework 4: daily return analysis using ARIMA and ARMA(0, 0) 
#### Dataset
`stock-treasury-2004_2006.csv`- contains the following:
* `TREAS_3M`: the yield of the 3-month treasury note in percent (i.e 2.1 means 2.1%)
* `Adjusted close price` of ten major stocks: GM, F, UTX, CAT, MRK, PFE, IBM, MSFT, C, XOM
* `SP`: The S&P 500 equity index level at the close of the trading day

#### Models
**ARIMA** and **white noise model ARMA(0, 0)**.

#### Precedure
* Fit **ARIMA** models on the DailyRet time series, up to AR order p=2, MA order q=2, and differencing order d=1.

* Display the summary of the best selected model based on the AIC criterion.
* Plot the original returns series and the predictions of the best selected model using the model's plot_predict method.
<img src="https://github.com/bondxue/FRE7773-Machine-Learning-in-Finance/blob/master/HW04-TimeSeries/images/return.PNG" width="700">

* Run the *Jarque-Bera normality test* on the residuals of the best selected **ARIMA** model, and produce the *qq plot* of the residuals.
```Python
Jarque-Bera Normality Test on AR Residuals
statistic: 1.5098564497740838
p-value: 0.4700443510061998
skew: -0.10713348436248081
kurtosis: 3.0895159804068806
```

<img src="https://github.com/bondxue/FRE7773-Machine-Learning-in-Finance/blob/master/HW04-TimeSeries/images/qq_arima.PNG" width="400">

* Repeat the *Jarque-Bera test* and the *qq plot* using the residuals of the **white noise model ARMA(0, 0)**.
```Python
Jarque-Bera Normality Test on AR Residuals
statistic: 0.6653429629453448
p-value: 0.717005705127056
skew: 0.02162822835081098
kurtosis: 3.1479565058733145
```
<img src="https://github.com/bondxue/FRE7773-Machine-Learning-in-Finance/blob/master/HW04-TimeSeries/images/qq_arma.PNG" width="400">

* Compare the two and comment on whether they are really different.


#### Results 
+ We use *Jarque-Bera* test to check the normality for rediduals of both **ARIMA(2,1,2)** and **white noise ARMA** model, we could see that *p-value* of **ARIMA(2,1,2)** (0.470) is smaller than **white noise ARMA(0,0)** model (0.717), which *means residual* of **white noise ARMA(0,0)** is normal with larger confidence than **ARIMA(2,1,2)** model.
+ **ARIMA(2,1,2)** has longer left tail, **while ARMA(0,0)**  has longer right tail based on skew value.
+ Based on *test statistic*,  residuel of **ARMA(0,0)** (0.665) is closer to a normal distribution than **ARIMA(2,1,2)** (1.509).
+ Based on *QQ plot*, they are both approximate to normal distrtribution
+ In conclusion, based on *QQ plot* and *Jarque-Bera test*, we can not reject that the residuals of of **ARIMA(2,1,2)** and **ARMA(0,0)** are normal. Therefore, I could say that these two models are not really different to each other.
+ It is unexpected since I thought **ARIMA(2,1,2)** may perform better than **white noise ARMA(0,0)**. The potential reason is that we need to choose a model in which at least one of p or q is no larger than 1 to reduce the complexity of the model, otherwise the model may lead to *overfitting* problem. 


---------------------------
### homework 5: daily return prediction using LSTM 

#### Dataset
`stock-treasury-2004_2006.csv`- contains the following:
* `TREAS_3M`: the yield of the 3-month treasury note in percent (i.e 2.1 means 2.1%)
* `Adjusted close price` of ten major stocks: GM, F, UTX, CAT, MRK, PFE, IBM, MSFT, C, XOM
* `SP`: The S&P 500 equity index level at the close of the trading day

#### Model
**RNN** with two **LSTM** cells

#### Precedure
1. Build an RNN with two LSTM cells and train it on the first 607 sequences. Use the remaining returns for out-of-sample testing.
2. Compute the out-of-sample actual and predicted 2-day, 3-day, ..., N-day return, by summing 1-day forward returns up to this horizon.
3. Calculate and report the RMSE and the correlation between actual and predicted 1-day, 2-day, ..., N-day returns.
```Python
1-day return RMSE: 79.28
2-day return RMSE: 114.62
3-day return RMSE: 144.13
4-day return RMSE: 173.44

1-day return correlation: -0.16756074223160694
2-day return correlation: -0.13458050383693185
3-day return correlation: -0.23897153449425
4-day return correlation: -0.26175980947060246
```
4. Plot the actual and predicted returns in the out-of-sample part. Here is 3-day return plot result:
<img src="https://github.com/bondxue/FRE7773-Machine-Learning-in-Finance/blob/master/HW05-LSTM/images/3-day%20return.PNG" width="800">


#### Summary
Based on *RMSE* and *correlation* between acutal and predicted n-day returns  and plots of out-of-sample parts, we can conclude that LSTM model does not have good performance in predicting daily returns. The potential reseaons are:
1. Daily return has **martingale** property, we cannot predict daily returns just based on the previous daily return data. 
2. If we want to improve the prediction performance, instead of using just previous daily return data, more features needed to be considered, such as `volume`.
3. Even with enough data and good feature selection, we still cannot expect much better performace. As we could see from the real daily return plots, daily return does not have obvious trends and it is extremly unstable with sudden nonlinear changes. However, LSTM only based on the previous information, for which I doubt it could capture the super nonlinear trends. 
