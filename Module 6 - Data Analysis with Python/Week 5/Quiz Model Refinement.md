### Quiz: Model Refinement
> 
> Total points 7
> 
> 1.
> 
> Question 1
> 
> What is the correct use of the "train_test_split" function such that 40% of the data samples will be utilized for testing, the parameter "random_state" is set to zero, and the input variables for the features and targets are x_data, y_data respectively?
> 
> 1 point
> 
>  train_test_split(x_data, y_data, test_size=0, random_state=0.4) 
> 

      train_test_split(x_data, y_data, test_size=0.4, random_state=0) 
> 
>  train_test_split(x_data, y_data) 
> 
> 2.
> 
> Question 2
> 
> What is the output of the following code?
> 
> 
>  cross_val_score(lre, x_data, y_data, cv=2)
> 
> 
> 1 point
> 

      The average R^2 on the test data for each of the two folds 
> 
>  This function finds the free parameter alpha 
> 
>  The predicted values of the test data using cross-validation 
> 
> 3.
> 
> Question 3
> 
> What is the output of the following code?
> 
> <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> 
> cross_val_predict (lr2e, x_data, y_data, cv=3)
> 
> 
> 1 point
> 

      The predicted values of the test data using cross-validation 
> 
>  The average R^2 on the test data for each of the two folds 
> 
>  This function finds the free parameter alpha 
> 
> 4.
> 
> Question 4
> 
> What dictionary value would we use to perform a grid search to determine if normalization should be used and for testing the following values of alpha? 1,10, 100
> 
> 1 point
>
> 
> {'alpha': [1,10,100]}]
> 
> alpha=[1,10,100]     
> normalize=[True,False]
> 
> 

     [{'alpha':[1,10,100],'normalize':[True,False]} ]
> 
> 
> 5.
> 
> Question 5
> 
> You train a ridge regression model, you get a R^2 of 1 on your training data and you get a R^2 of 0 on your validation data; what should you do?
> 
> 1 point
> 

      Your model is overfitting, so increase the parameter alpha 
> 
>  Your model is under fitting; so perform a polynomial transform 
> 
>  Nothing, your model performs flawlessly on your validation data 
> 
> 6.
> 
> Question 6
> 
> The following is an example of what?
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bQe6jTd-EeiXeAoF3sTtsA_a2e60ade500ba6d3b93fe11037f0593d_Screen-Shot-2018-04-03-at-4.34.36-PM.png?expiry=1597622400000&hmac=YubBozJjO5yxoxqB1eXx8ei-H9LoO9pFQa99CIIK-cI)
> 
> 1 point
> 

      Overfitting 
> 
>  Perfect fit 
> 
>  Underfitting 
> 
> 7.
> 
> Question 7
> 
> The following is an example of what?
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/SBkr8zeEEeissBKg1xWY5A_00227225a1f9f10045c83602ccef31e7_Screen-Shot-2018-04-03-at-5.17.08-PM.png?expiry=1597622400000&hmac=aL5xIyVx4pKRaq4xCKQeLaxemx2Tagi4OlXCO1xgDzU)
> 
> 1 point
> 
>  Overfitting 
> 
>  Perfect fit 
> 

      Underfitting
>
> -- https://www.coursera.org/learn/data-analysis-with-python/exam/WAqqP/quiz-model-refinement/attempt#Tunnel Vision Close
