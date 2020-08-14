### Measures for In-Sample Evaluation
> 
> Total points 2
> 
> 1.
> 
> Question 1
> 
> Of the following answer values, which one is the minimum value of **R^2**?
> 
> 1 point
> 
>  10 
> 

      0 
> 
>  1 
> 
> 2.
> 
> Question 2
> 
> Select the correct function to calculate the mean squared error between yhat and y
> 
> 1 point
> 
> lm = LinearRegression()
> 
> lm.score(X,y)
> 
> X = df[['highway-mpg']]
> 
> Y = df['price']
> 
> lm.fit(X, Y)
> 
> out=lm.score(X,y)
>

     from  sklearn.metrixs import mean_squared_error
     
     mean_squared_error(y,yhat)
> 
>
> -- https://www.coursera.org/learn/data-analysis-with-python/exam/JJI1r/measures-for-in-sample-evaluation/attempt#Tunnel Vision Close
