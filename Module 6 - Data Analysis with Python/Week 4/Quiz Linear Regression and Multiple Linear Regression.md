### Linear Regression and Multiple Linear Regression
> 
> Total points 2
> 
> 1.
> 
> Question 1
> 
> consider the following lines of code, what variable contains the predicted values :
>

     from sklearn.linear_model import LinearRegression 
     
     lm=LinearRegression()
     
     X = df[['highway-mpg']]
     
     Y = df['price']
     
     lm.fit(X, Y)
     
     Yhat=lm.predict(X)
> 
> 
> 1 point
> 
>  X 
> 

      Yhat 
> 
>  Y 
> 
> 2.
> 
> Question 2
> 
> consider the following equation:
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/mrc50jdfEeiKpA6ZQCE7wA_d5b99b98180e37f67b0659d10a8104f5_eq.png?expiry=1597536000000&hmac=IupYCKH_LX7twU7qthnT64-lhwtsNIZBjfiUChmGPtE)
> 
> the variable y is?
> 
> 1 point
> 

      the target or dependent variable 
> 
>  the predictor or independent variable 
> 
>  the intercept
>
> -- https://www.coursera.org/learn/data-analysis-with-python/exam/nxJgV/linear-regression-and-multiple-linear-regression/attempt#Tunnel Vision Close
