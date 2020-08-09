## Data Wrangling
> 
> Total points 6
> 
>  1.Question 1
> 
> What task does the following line of code perform?
> 
> 
> df['peak-rpm'].replace(np.nan, 5,inplace=True)
> 
> 1 point 
> 

      replace the not a number values with 5 in the column 'peak-rpm' 
> 
>  rename the column 'peak-rpm' to 5 
> 
>  add 5 to the dataframe **df** 
> 
>  2.Question 2
> 
> What task do the following lines of code perform?
> 
> 
> avg=df['horsepower'].mean(axis=0)
> 
> df['horsepower'].replace(np.nan, avg)
> 
> 1 point 
> 
>  replace all the NaN values with the mean 
> 

      nothing; because the parameter **inplace** is not set to true 
> 
>  calculate the mean value for the **'horsepower'** column and replace all the NaN values of that column by the mean value 
> 
>  3.Question 3
> 
> Consider the dataframe df; convert the column df["city-mpg"] to df["city-L/100km'] by dividing 235 by each element in the column 'city-mpg'.
> 
> 1 point 
> 

     df['city-L/100km'] = 235/df["city-mpg"]
> 
> 
> df['city-L/100km'] = df["city-mpg"].diV(235)
> 
> </pre> 
> 
>  4.Question 4
> 
> What data type is the following set of numbers? **666, 1.1,232,23.12**
> 
> 1 point 
> 
>  int 
> 

      float 
> 
>  object 
> 
>  5.Question 5
> 
> The following code is an example of:
> 
> df['length'] = df['length']/df['length'].max()
> 
> 1 point 
> 

      simple feature scaling 
> 
>  min-max scaling 
> 
>  z-score 
> 
>  6.Question 6
> 
> Consider the two columns 'horsepower', and 'horsepower-binned'; from the dataframe **df**; how many categories are there in the 'horsepower-binned' column?
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/HNPUdjOYEeiP-Qrke_kVoA_5cd6eb4e5587dc6cbb6a6ed79a95f98c_Screen-Shot-2018-03-29-at-5.29.19-PM.png?expiry=1597104000000&hmac=28bJOK0RSfxN-ti3BoK7A1008GgTR8xEFvBfSC2zqgg)
> 
> 1 point 
> 
> Enter answer here
>

    3
>
> -- https://www.coursera.org/learn/data-analysis-with-python/exam/1zQMz/data-wrangling/attempt#Tunnel Vision Close
