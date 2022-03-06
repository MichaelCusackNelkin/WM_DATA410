# Project 4: Multiple Boosting Algorithms and Light Gradient Boosting Machines (LightGBM)

## Abstract
In this paper I compare different combinations of regression models and gradient boosting models, and asses their performance on the Concrete Compressive Strength dataset provided by USCI. I use a home-made repetitive gradient boosting algorithm on different combinations of Random Forest, Loess, and Decision Tree regressors. Here, I omit Neural Networks since their regression performance tends to be poor, and they are resource intensive.

I also investiagate and apply LightGBM, Microsoft's resource-light gradient boosting machine. In the end, LightGBM out-performs my home-made algorithm with an average MSE of 108 versus my home made's 148.

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
This function is able to use Loess, Random Forests, or Decision trees as both 'weak learners' and as gradient boosters. It works by a series of conditionals to figure out what the user would like to do, then training a weak learner and using a for loop to train a number of boosting algorithms for as many iterations as are input by the user.
## LightGBM
LightGBM is a powerful gradient boosting algorithm that uses decision tree boosters. The thing that makes LightGBM different is that it grows its trees by each leaf, rather than by each level.  This means that for each decision tree, the only leaf that gets to split into the next level is the on with the highest loss. This is great for using fewer resources on large datasets, and for processing speed, however it can lead to overfitting on small datasets. This change is why LightGBM has 'light' in its name, as it used much less memory than other competitive algorithms and gives results quickly.
## Performance Conclusion
Here, I used KFold validation with 10 splits, and only tried one random state to try out. This validation loop took much longer than others that I've used, it took well over an hour to train and I had to use my desktop GPU acceleration. Below si my validation loop:

```
scale = StandardScaler()
for i in range(12345,12346):
  print('Random State: ' + str(i))
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
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
    lgb = lgbm.LGBMRegressor(num_iterations=200)
    lgb.fit(xtrain, ytrain, eval_set=[(xtest, ytest)], eval_metric='mse')
    yhat_lgbm = lgb.predict(xtest, num_iteration=lgb.best_iteration_)


    #Append each model's MSE
    mse_lwr_lwr.append(mse(ytest,yhat_lwr_lwr))
    mse_lwr_d.append(mse(ytest,yhat_lwr_d))
    mse_lwr_rf.append(mse(ytest,yhat_lwr_rf))
    mse_rf_d.append(mse(ytest,yhat_rf_d))
    mse_lgbm.append(mse(ytest,yhat_lgbm))
```
