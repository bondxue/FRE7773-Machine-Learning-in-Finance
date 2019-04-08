import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

def fit_transform_ohe(df, col_name):
    """This function performs one hot encoding for the specified column.

    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        col_name: the column to be one hot encoded

    Returns:
        tuple: label_encoder, one_hot_encoder, transformed column as pandas DataFrame

    """
    # label encode the column
    le = preprocessing.LabelEncoder()
    le_labels = le.fit_transform(df[col_name])
    df[col_name+'_label'] = le_labels
    
    # one hot encoding
    ohe = preprocessing.OneHotEncoder(categories='auto')
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    
    return le, ohe, features_df


def transform_ohe(df, le, ohe, col_name):
    """This function performs one hot encoding for the specified
        column using the specified label and one hot encoder objects.

    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        le(Label Encoder): the label encoder object used to fit label encoding
        ohe(One Hot Encoder): the one hot encoder object used to fit one hot encoding
        col_name: the column to be one hot encoded

    Returns:
        tuple: transformed column as pandas DataFrame

    """
    # label encoder
    col_labels = le.transform(df[col_name])
    df[col_name+'_label'] = col_labels
    
    # ohe 
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    
    return features_df


def plot_learning_curves(model, X, y, ax, test_size=0.2, step=1, random_state=None):
    """This function fits the model to the data using CV, computes and plots 
		the resulting learning curves. The metric used is RMSE
		
    Args:
        model:  a model that conforms to the sklearn interface
        X: 		an indexable array with the features
        y:      an indexable array with the output variable, of same length as X
        ax:     the axis object on which to plot        
        test_size: float, int or None; 
				if float, should be between 0.0 and 1.0 and represents the test size proportion
				if int, it represents the absolute number of test samples
        step:   int; step size to iterate over the training sample, from 1 to len(X_train)
        random_state: random_state to use for train_validation split

    Returns:
		tuple: two arrays with the train metric and the validation metric 
          	   for each sample size in the range (1, len(X_train + 1, step)
	"""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    train_rmse, val_rmse = [], []
    
    for m in range(1, len(X_train) + 1, step):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_rmse.append(np.sqrt(mean_squared_error(y_train_predict, y_train[:m])))
        val_rmse.append(np.sqrt(mean_squared_error(y_val_predict, y_val)))
    
    ax.plot(train_rmse, "r-.", label="Training Set")
    ax.plot(val_rmse, "b--", label="Validation Set")
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('RMSE')
    ax.legend()

    return train_rmse, val_rmse

def plot_svc_decision_boundary(svc_model, ax, xmin, xmax):
    """This function plots the decision boundary of an SVC model and marks the support vectors.
        It uses only the first two features.
        Adapted from Aurelien Geron's notebook.

    Args:
        svc_model:  an instance of sklearn.svm.SVC fitted model
        ax:         the axis object on which to plot        
        xmin:       the minimum x-value
        xmax:       the maximum x-value
    """
    w = svc_model.coef_[0]
    b = svc_model.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0  => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    slab_up = decision_boundary + margin
    slab_down = decision_boundary - margin

    supp_vec = svc_model.support_vectors_
    ax.scatter(supp_vec[:, 0], supp_vec[:, 1], s=180, facecolors='#FFFFAA')
    ax.plot(x0, decision_boundary, "k-", linewidth=2)
    ax.plot(x0, slab_up, "k--", linewidth=2)
    ax.plot(x0, slab_down, "k--", linewidth=2)


def plot_predictions(svc_model, ax, axlim):
    """This function computes and plots the predictions of an SVC model.
        It uses only the first two features, x0 and x1.
        Adapted from Aurelien Geron's notebook.

    Args:
        svc_model:  an instance of sklearn.svm.SVC or LinearCVC fitted model
        ax:         the axis object on which to plot        
        axlim:      [xmin, xmax, ymin, ymax]
    """
    x0s = np.linspace(axlim[0], axlim[1], 100)
    x1s = np.linspace(axlim[2], axlim[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = svc_model.predict(X).reshape(x0.shape)
    y_decision = svc_model.decision_function(X).reshape(x0.shape)
    ax.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    ax.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

def plot_classes(X, y, ax, axlim):
    """This function plots the features and colors the points based on the corresponding classes.
        It uses only the first two columns of X.

    Args:
        X: 		    an indexable array with the features
        y:          an indexable array with the output classes coded as 0 or 1, of same length as X        
        ax:         the axis object on which to plot
        axlim:      [xmin, xmax, ymin, ymax]
    """
    ax.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    ax.plot(X[:, 0][y==1], X[:, 1][y==1], "go")
    ax.axis(axlim)
    ax.grid(True, which='both')


def plot_decision_boundary(model, ax, axlim, color_map='PuBuGn'):
    """This function plots the decision boundaries of a model as filled contours in 2D.
        It uses only the first two features, x0 and x1.

    Args:
        model:      an instance of sklearn fitted model
        ax:         the axis object on which to plot        
        axlim:      [xmin, xmax, ymin, ymax]
        color_map:  name of the color map to use
    """
    x1s = np.linspace(axlim[0], axlim[1], 100)
    x2s = np.linspace(axlim[2], axlim[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = model.predict(X_new).reshape(x1.shape)
    cmap = plt.cm.get_cmap(color_map)
    ax.contourf(x1, x2, y_pred, alpha=0.3, cmap=cmap)
    ax.axis(axlim)


def plot_regression(reg_model, x, y, ax, axlim):
    """This function plots the observations and predictions of regression model with one predictor

    Args:
        reg_model:  an instance of sklearn regression model
        x:          array with the predictor
        y:          array with the output
        ax:         the axis object on which to plot        
        axlim:      [xmin, xmax, ymin, ymax]
    """    
    x1 = np.linspace(axlim[0], axlim[1], 500).reshape(-1, 1)
    y_pred = reg_model.predict(x1)
    ax.axis(axlim)
    ax.plot(x, y, "b.")
    ax.plot(x1, y_pred, "r.-", linewidth=2)


def run_adf_test(ts, verbose=True):
    """Runs the augmented Dickey-Fuller test and returns statistics. 
       Adapted from https://github.com/dipanjanS/practical-machine-learning-with-python

    Args:
        ts:         time series object
        verbose:    True for printing the results on screen

    Returns:
		pd.Series: vector with statistics
    """
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['statistic',
                                             'p-value',
                                             'n_lags',
                                             'n_obs'])
    for key,value in dftest[4].items():
        dfoutput['critical_Val (%s)'%key] = value
    print(dfoutput)


def auto_arima(ts, p_max=1, q_max=1, d_max=1, verbose=True):
    """ Scans a grid of ARIMA(p, d, q) parameters, and returns the best model based on the AIC criterion.
       Adapted from https://github.com/dipanjanS/practical-machine-learning-with-python
       The best model is selected based on minimum AIC criterion

    Args:
        ts:         time series object
        p_max:      maximum AR(p) parameter to search for; must be a non-negative integer
        q_max:      maximum MA(q) parameter to search for; must be a non-negative integer
        d_max:      maximum ARIMA(, d, ) parameter to search for; integer between 0 and 2 inclusive
        verbose:    True for printing the results on screen

    Returns:
		tuple:      (best_model, model_results) 
                    best_model is a dictionary with the best model object, params, AIC and BIC
                    model_results is a list of dictionaries as above, for all fitted models
    """    
    # ARIMA parameters p, d and q parameters can take any value between 0 and par_max
    p_range = range(0, p_max + 1)
    q_range = range(0, q_max + 1)
    d_range = range(0, min(d_max + 1, 3))

    # Generate all possible triplets of parameters
    # p = q = range(0, par_max + 1)
    pdq = [(x[0], x[1], x[2]) for x in list(itertools.product(p_range, d_range, q_range))]
    
    model_results = []
    best_model = {}
    min_aic = 1000000
    for params in pdq:
        try:
            mod = sm.tsa.ARIMA(ts, order=params)
            results = mod.fit()
            
            if verbose:
                print('ARIMA{0} AIC:{1:.2f}  BIC:{2:.2f}'.format(params, results.aic, results.bic))
            model_results.append({'aic':results.aic,
                                  'bic':results.bic,
                                  'params':params,
                                  'model':results})
            if min_aic > results.aic:
                best_model={'aic':results.aic,
                            'bic':results.bic,
                            'params':params,
                            'model':results}
                min_aic = results.aic
        except Exception as ex:
            print(ex)
    if verbose:
        print('Best model params:{0} AIC:{1:.2f}  BIC:{2:.2f}'.format(best_model['params'],
              best_model['aic'], best_model['bic']))  

    return best_model, model_results
