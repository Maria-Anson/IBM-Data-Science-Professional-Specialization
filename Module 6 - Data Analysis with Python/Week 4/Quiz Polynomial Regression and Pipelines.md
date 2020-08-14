### Polynomial Regression and Pipelines
> 
> Total points 2
> 
> 1.
> 
> Question 1
> 
> What functions are used to generate Polynomial Regression with more than one dimension
> 
> 1 point
> 
> f=np.polyfit(x,y,3)
> 
> p=np.poly1d(f)
> 

     pr=PolynomialFeatures(degree=2)
     
     pr.fit_transform([1,2], include_bias=False)
> 
> 
> 2.
> 
> Question 2
> 
> select the line of code that imports the module required for polynomial features
> 
> 1 point
> 
>  <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> 
> from sklearn.linear_model import LinearRegression
> 

     from sklearn.preprocessing import PolynomialFeatures
> 
> from sklearn.preprocessing import StandardScaler
> 
>
> -- https://www.coursera.org/learn/data-analysis-with-python/exam/LehAa/polynomial-regression-and-pipelines/attempt#Tunnel Vision Close
