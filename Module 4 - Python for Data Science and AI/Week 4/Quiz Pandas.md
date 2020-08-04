# Pandas
> 
> Total points 3
> 
>  1.Question 1
> 
> how would you select the columns with the headers: artist, length and genre from the dataframe **df** and assign them to the variable **y**
> 
> 1 point 
> 

      y=df[['Artist','Length','Genre']] 
> 
>  y=df['Artist','Length','Genre'] 
> 
>  y=df[['Artist'],['Length'],['Genre']] 
> 
>  2.Question 2
> 
> consider the dataframe **df** how would you access the element in the 2nd row and 1st column
> 
> 1 point 
> 

      df.ix[1,0] 
> 
>  df.ix[2,1] 
> 
>  df.ix[0,1] 
> 
>  3.Question 3
> 
> what function would you use to load a csv file in Pandas
> 
> 1 point 
> 

      pd.read_csv 
> 
>  pd.read_excel
>
> -- https://www.coursera.org/learn/python-for-applied-data-science-ai/exam/J2snC/pandas/attempt#Tunnel Vision Close
