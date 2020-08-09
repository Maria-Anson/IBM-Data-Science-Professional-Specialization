## Data Formatting in Python
> 
> Total points 2
> 
>  1.Question 1
> 
> How would you add one to each element in the column **df["a"]** and assign it back to the column **df["a"]**?
> 
> 
>  df["a"]+1
> 

      df["a"]=df["a"]+1
> 
>  you can't do this in Python 
> 
>  2.Question 2
> 
> How would you cast each element in the column **"price"** to an integer?
> 
> 1 point 
> 
> df["price"] = int(df["price"])
> 

     df["price"] = df["price"].astype("int")
> 
>
> -- https://www.coursera.org/learn/data-analysis-with-python/exam/378Kn/data-formatting-in-python/attempt#Tunnel Vision Close
