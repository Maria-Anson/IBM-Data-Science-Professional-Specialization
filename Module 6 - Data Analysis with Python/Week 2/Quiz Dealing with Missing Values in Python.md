## Dealing with Missing Values in Python
> 
> Total points 3
> 
>  1.Question 1
> 
> How would you access the column "symboling” from the dataframe **df?**
> 
> 1 point 
>
> 

     df["symboling"] 
> 
> df=="symboling"
> 
> 
> </pre> 
> 
>  2.Question 2
> 
> How would you replace the missing values in the column "normalized-losses" with the mean of that column?
> 
> 1 point 
> 
> 

     mean = df["normalized-losses"].mean() 
     
     df["normalized-losses"].replace(np.nan, mean)
> 
> df.dropna(subset=["price"], axis=0, inplace = True)
> 
>  3.Question 3
> 
> What is the correct symbol for missing data?
> 
> 1 point 
> 

      nan 
> 
>  no-data
>
> -- https://www.coursera.org/learn/data-analysis-with-python/exam/t0hOj/dealing-with-missing-values-in-python/attempt#Tunnel Vision Close
