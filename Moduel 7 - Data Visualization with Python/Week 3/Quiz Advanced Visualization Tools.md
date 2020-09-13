## Advanced Visualization Tools
> 
> Latest Submission Grade
> 
> 100%
> 
> 1.
> 
> Question 1
> 
> Seaborn is a Python visualization library that provides a high-level interface for visualizing geospatial data.
> 
> 1 / 1 point
> 
>  True. 
> 

      False. 
> 
> Check
> 
> Correct
> 
> Correct.
> 
> 2.
> 
> Question 2
> 
> Which of the choices below will create the following regression line plot, given a _pandas_ dataframe, **data_df**?
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/f8IwPThNEeizZhIWROSU4A_b1d5c41b1905fbe55be358a0305d4492_Screen-Shot-2017-07-24-at-9.25.35-AM.png?expiry=1600128000000&hmac=0cq6T28qC8Mxv9xBPMQYTfVFy-3W-3twGrK2ACIgJLg)
> 
> 1 / 1 point
> 
> data_df.plot(kind="regplot", color="green", marker="+")
> 
> 
> data_df.plot(kind="regression", color="green", marker="+")
> 
> 

    > import seaborn as sns
     
    > ax = sns.regplot(x="year", y="total", data=data_df, color="green", marker="+")
> 
> 
> import seaborn as sns
> 
> ax = sns.regplot(x="total", y="year", data=data_df, color="green")
> 
> 
> import seaborn as sns
> 
> ax = sns.regplot(x="year", y="total", data=data_df, color="green")
> 
> 
> Check
> 
> Correct
> 
> Correct.
> 
> 3.
> 
> Question 3
> 
> A waffle chart is a great way to visualize data in relation to a whole, or to highlight progress against a given threshold.
> 
> 1 / 1 point
> 

      True. 
> 
>  False. 
> 
> Check
> 
> Correct
> 
> Correct.
> 
> 4.
> 
> Question 4
> 
> The easiest way to create a waffle chart in Python is using the Python package, PyWaffle.
> 
> 1 / 1 point
> 

      True. 
> 
>  False. 
> 
> Check
> 
> Correct
> 
> Correct.
> 
> 5.
> 
> Question 5
> 
> A word cloud (choose all that apply)
> 
> 1 / 1 point
> 
>  can be easily created using Matplotlib using the scripting layer. 
> 
>  is a depiction of the frequency of the stopwords, such as a, the, and, in some textual data. 
> 

      is a depiction of the meaningful words in some textual data, where the more a specific word appears in the text, bigger and bolder it appears in the word cloud. 
> 
> Check
> 
> Correct
> 
> Correct.
> 

      is a depiction of the frequency of different words in some textual data. 
> 
> Check
> 
> Correct
> 
> Correct.
> 

      can be generated in Python using the _word_cloud_ package that was developed by **Andreas Mueller**. 
> 
> Check
> 
> Correct
> 
> Correct.
>
> -- https://www.coursera.org/learn/python-for-data-visualization/exam/nJc7I/advanced-visualization-tools/attempt?redirectToCover=true#Tunnel Vision Close
