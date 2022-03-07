# Project 4: Multiple Boosting Algorithms and Light Gradient Boosting Machines (LightGBM)

## Abstract
In this paper I compare different combinations of regression models and gradient boosting models, and asses their performance on the Concrete Compressive Strength dataset provided by USCI. I use a home-made repetitive gradient boosting algorithm on different combinations of Random Forest, Loess, and Decision Tree regressors. Here, I omit Neural Networks since their regression performance tends to be poor, and they are resource intensive.

I also investiagate and apply LightGBM, Microsoft's resource-light gradient boosting machine. In the end, LightGBM out-performs my home-made algorithm with an average MSE of 108 versus my home made's 148, but only when the entire dataset is used (80% train, 20% test). Otherwise, during KFold validations, the best model was Loess with three repeated RandomForest boosters. I believe this difference in performance has to do with LightGBM's overfitting problem on small datasets. The entire concrete compressive strength dataset has just 1048 observations, so splitting it during KFold makes it much too small for LightGBM.

## Methods
In this paper I designed a home made gradient boosting algorithm that could mix and match Decision Trees, Random Forests, and Loess regressors. The Loess regressor was also homemade. below is the code for the function,

```
def n_boost(X, y, xtest, model, nboost, booster, kern = None, tau = None, tau_b = None, 
            intercept = None, n_estimators=None , max_depth=None, model_nn = None):
  if booster == 'LWR':
    if model == 'LWR':
      yhat = lw_reg(X,y,X,kern,tau,intercept) #get loess predictions on training data
      yhat_test = lw_reg(X,y,xtest,kern,tau,intercept) #get loess predictions on testing data
      lw_error = y - yhat #find the loess training residuals; these are what the booster will train on
      for i in range(nboost): 
        yhat += lw_reg(X,lw_error,X,kern,tau_b,intercept)
        yhat_test += lw_reg(X,lw_error,xtest,kern,tau_b,intercept)
        lw_error = y - yhat
      return yhat_test

    if model == 'RF' or model == 'RFR':
      model_rf = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth)
      model_rf.fit(X,y)
      yhat_rf = model_rf.predict(X)
      yhat_test = model_rf.predict(xtest)
      rf_error = y - yhat_rf
      for i in range(nboost): 
        yhat_rf += lw_reg(X,rf_error,X,kern,tau_b,intercept)
        yhat_test += lw_reg(X,rf_error,xtest,kern,tau_b,intercept)
        rf_error = y - yhat_rf
      return yhat_test

  else:

    if model == 'LWR':
      yhat = lw_reg(X,y,X,kern,tau,intercept) #get loess predictions on training data
      yhat_test = lw_reg(X,y,xtest,kern,tau,intercept) #get loess predictions on testing data
      lw_error = y - yhat #find the loess training residuals; these are what the booster will train on
      for i in range(nboost): 
        booster.fit(X, lw_error)
        yhat += booster.predict(X)
        yhat_test += booster.predict(xtest)
        lw_error = y - yhat
      return yhat_test

    if model == 'RF' or model == 'RFR':
      model_rf = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth)
      model_rf.fit(X,y)
      yhat_rf = model_rf.predict(X)
      yhat_test = model_rf.predict(xtest)
      rf_error = y - yhat_rf
      for i in range(nboost): 
        booster.fit(X, rf_error)
        yhat_rf += booster.predict(X)
        yhat_test += booster.predict(xtest)
        rf_error = y - yhat_rf
      return yhat_test
```
This function is able to use Loess, Random Forests, or Decision trees both as 'weak learners' and as gradient boosters. It works via series of conditionals that figure out what the user would like to do, and then by training a weak learner and using it for a loop to train a number of boosting algorithms for as many iterations as are input by the user.
## LightGBM
LightGBM is a powerful gradient boosting algorithm that uses decision tree boosters. The thing that makes LightGBM different is that it grows its trees by each leaf, rather than by each level.  This means that for each decision tree, the only leaf that gets to split into the next level is the on with the highest loss. This is great for using fewer resources on large datasets, and for processing speed, however it can lead to overfitting on small datasets. This change is why LightGBM has 'light' in its name, as it used much less memory than other competitive algorithms and gives results quickly.

LightGBM's implementation is also strikingly simple:
```
lgb = lgbm.LGBMRegressor(num_iterations=1000)
    lgb.fit(xtrain, ytrain, eval_set=[(xtest, ytest)], eval_metric='mse', early_stopping_rounds=100)
    yhat_lgbm = lgb.predict(xtest, num_iteration=lgb.best_iteration_)
```
## Performance Conclusion
Here, I used KFold validation with 10 splits, and only tried one random state to try out. This validation loop took much longer than others that I've used, it took well over an hour to train and I had to use my desktop GPU acceleration. Below is my validation loop:

```
for i in range(123115,123117):
  print('Random State: ' + str(i))
  kf = KFold(n_splits=5,shuffle=True,random_state=i)
  # this is the random state cross-validation loop to make sure our results are real, not just the state being good/bad for a particular model
  j = 0
  for idxtrain, idxtest in kf.split(data[:,:2]):
    t = time.time()
    j += 1
    #Split the train and test data
    xtrain = data[:,:6][idxtrain]
    ytrain = data[:,-1][idxtrain]
    ytest = data[:,-1][idxtest]
    xtest = data[:,:6][idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)


    #LWR boosted with decision tree
    booster = DecisionTreeRegressor(max_depth=2)
    yhat_lwr_d = n_boost(xtrain, ytrain, xtest, model = 'LWR', nboost=3, booster=booster, 
                   kern = Tricubic, tau = 1.2, intercept = True,)
    
    #LWR boosted with RF
    booster = RandomForestRegressor(n_estimators=25, max_depth=2)
    yhat_lwr_rf = n_boost(xtrain, ytrain, xtest, model = 'LWR', nboost=3, booster=booster, 
                   kern = Tricubic, tau = 1.2, intercept = True)

    #LWR boosted with LWR
    booster='LWR'
    yhat_lwr_lwr = n_boost(xtrain, ytrain, xtest, model = 'LWR', nboost=3, booster=booster, 
                   kern = Tricubic, tau = 1.2, tau_b=0.5, intercept = True)
    
    #RF boosted with decision tree
    booster = DecisionTreeRegressor(max_depth=2)
    yhat_rf_d = n_boost(xtrain, ytrain, xtest, model = 'RFR', nboost=3, booster=booster, n_estimators=100 , max_depth=3)
    
    #RF boosted with LWR
    booster = 'LWR'
    yhat_rf_d = n_boost(xtrain, ytrain, xtest, model = 'RFR', nboost=3, booster=booster, 
                   kern = Tricubic, tau = 1.2, tau_b=0.5, intercept = True, n_estimators=100 , max_depth=3)

    #LightGBM
    lgb = lgbm.LGBMRegressor(num_iterations=1000)
    lgb.fit(xtrain, ytrain, eval_set=[(xtest, ytest)], eval_metric='mse', early_stopping_rounds=100)
    yhat_lgbm = lgb.predict(xtest, num_iteration=lgb.best_iteration_)
```

After running this validation loop, my results were:

The Cross-validated Mean Squared Error for LWR with Decision Tree is : 143.012

>The Cross-validated Mean Squared Error for LWR with Random Forest is : 140.36

>The Cross-validated Mean Squared Error for Random Forest with Decision Tree is : 181.19

>The Cross-validated Mean Squared Error for LWR with LWR : 199.80

>The Cross-validated Mean Squared Error for LightGBM : 142.94

Surprisingly, LightGBM did not take the top spot. Instead Loess boosted by three repetitive Random Forests was the best performer with an average MSE of 143.51, while LightGBM ended with an average of 182.67. I believe this is because the dataset is small enough (1048 entries) that when it was split for training and testing, and further during KFold validation, it became much too small for LightGBM, which has an overfitting probelem for small datasets.

<figure>
<center>
<img src='Data/Proj4_MSEs.png' width='1600px' />
<figcaption>MSE Score per K Fold for Concrete Compressive Strength<figcaption></center>
</figure>