## Specialized Visualization Tools
> 
> Latest Submission Grade
> 
> 100%
> 
> 1.
> 
> Question 1
> 
> Bar charts are less confusing than pie charts and should be your first attempt when creating a visual to explore a dataset.
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
> 2.
> 
> Question 2
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/uYkg-DhJEeizZhIWROSU4A_1ed0d92500c6ce66e00996a01b5aa592_boxplot.png?expiry=1600128000000&hmac=t30QZMWJFxHk7wDhACivPnFFS0uHzHn3j0v1mAL6yu8)
> 
> What do the letters in the box plot above represent?
> 
> 1 / 1 point
> 
>  A = Mean, B = Upper Mean Quartile, C = Lower Mean Quartile, D = Inter Quartile Range, E = Minimum, and F = Outliers 
> 
>  A = Mean, B = Third Quartile, C = First Quartile, D = Inter Quartile Range, E = Minimum, and F = Outliers 
> 
>  A = Mean, B = Third Quartile, C = First Quartile, D = Inter Quartile Range, E = Minimum, and F = Maximum 
> 
>  A = Median, B = Third Quartile, C = Mean, D = Inter Quartile Range, E = Lower Quartile, and F = Outliers 
> 

      A = Median, B = Third Quartile, C = First Quartile, D = Inter Quartile Range, E = Minimum, and F = Outliers 
> 
> 
> Correct
> 
> 
> 3.
> 
> Question 3
> 
> What is the correct combination of function and parameter to create a box plot in Matplotlib?
> 
> 1 / 1 point
> 
>  Function = boxplot, and Parameter = type with value = "plot" 
> 
>  Function = box, and Parameter = type with value = "plot" 
> 
>  Function = plot, and Parameter = kind with value = "boxplot" 
> 

      Function = plot, and Parameter = kind with value = "box" 
> 
>  Function = plot, and Parameter = type with value = "box" 
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
> Which of the lines of code below will create the following scatter plot, given the _pandas_ dataframe, df_total?
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/XlfvbDwOEeiUEwpDBScAAA_3f3f034b93daff111b04b102b76f4926_scatter_plot.png?expiry=1600128000000&hmac=oLs4obAAouPjqagO3Pry5YKXnCCMEHuUb8yU4Fxk52E)
> 
> 1 / 1 point
> 
> import matplotlib.pyplot as plt
> 
> df_total.plot(type='scatter', x='year', y='total')
> 
> plt.title('Total Immigrant population to Canada from 1980 - 2013')
> 
> plt.label ('Year')
> 
> plt.label('Number of Immigrants')
> 
> 

    > import matplotlib.pyplot as plt
     
    > df_total.plot(kind='scatter', x='year', y='total')
     
    > plt.title('Total Immigrant population to Canada from 1980 - 2013')
     
    > plt.xlabel ('Year')
     
    > plt.ylabel('Number of Immigrants')
> 
> import matplotlib.pyplot as plt
> 
> plot(kind='scatter', x='year', y='total', data=df_total)
> 
> plt.title('Total Immigrant population to Canada from 1980 - 2013')
> 
> plt.label ('Year')
> 
> plt.label('Number of Immigrants')
> 
> 
> import matplotlib.scripting.pyplot as plt
> 
> df_total.plot(type='scatter', y='year', x='total')
> 
> plt.title('Total Immigrant population to Canada from 1980 - 2013')
> 
> plt.xlabel ('Year')
> 
> plt.ylabel('Number of Immigrants')
> 
> 
> import matplotlib.scripting.pyplot as plt
> 
> df_total.plot(kind='scatter', x='year', y='total')
> 
> plt.title('Total Immigrant population to Canada from 1980 - 2013')
> 
> plt.label('Year')
> 
> plt.label('Number of Immigrants')
> 
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
> A bubble plot is a variation of the scatter plot that displays three dimensions of data.
> 
> 1 / 1 point
> 

      True. 
> 
>  False 
> 
> Check
> 
> Correct
> 
> Correct.
>
> -- https://www.coursera.org/learn/python-for-data-visualization/exam/oQXsZ/specialized-visualization-tools/attempt?redirectToCover=true#Tunnel Vision Close
