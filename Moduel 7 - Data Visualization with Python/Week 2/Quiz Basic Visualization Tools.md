### Basic Visualization Tools
> 
> Latest Submission Grade
> 
> 100%
> 
> 1.
> 
> Question 1
> 
> Area plots are unstacked by default.
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
> The following code uses the artist layer to create a stacked area plot of the data in the _pandas_ dataframe, **area_df.**
> 
> import matplotlib.pyplot as plt
> 
> area_df.plot(kind='area', figsize=(20, 10))
> 
> plt.title('Plot Title')
> 
> plt.ylabel('Vertical Axis Label')
> 
> plt.xlabel('Horizontal Axis Label')
> 
> plt.show()
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
> 3.
> 
> Question 3
> 
> Which of the following codes will create an unstacked area plot of the data in the _pandas_ dataframe, **area_df**, with a transparency value of 0.35?
> 
> 1 / 1 point
> 
> 
> import matplotlib.pyplot as plt
> 
> transparency = 1 - 0.35 
> 
> area_df.plot(kind='area', alpha=transparency, stacked=False, figsize=(20, 10))
> 
> plt.title('Plot Title')
> 
> plt.ylabel('Vertical Axis Label')
> 
> plt.xlabel('Horizontal Axis Label')
> 
> plt.show()
> 
> 
> transparency = 0.35
> 
> ax = area_df.plot(kind='area', alpha=transparency, stacked=False, figsize=(20, 
> 
>   10))
> 
> ax.title('Plot Title')
> 
> ax.ylabel('Vertical Axis Label')
> 
> ax.xlabel('Horizontal Axis Label')
> 
> 
> transparency = 0.35
> 
> ax = area_df.plot(type='area_plot', alpha=transparency, stacked=False, figsize
> 
>   =(20, 10))
> 
> ax.title('Plot Title')
> 
> ax.ylabel('Vertical Axis Label')
> 
> ax.xlabel('Horizontal Axis Label')
> 
> 
> import matplotlib.pyplot as plt
> 
> area_df.plot(kind='area', stacked=False, figsize=(20, 10))
> 
> plt.title('Plot Title')
> 
> plt.ylabel('Vertical Axis Label')
> 
> plt.xlabel('Horizontal Axis Label')
> 
> plt.show()
> 
> 

      > transparency = 0.35
       
      > ax = area_df.plot(kind='area', alpha=transparency, stacked=False, figsize=(20, 10))
       
      > ax.set_title('Plot Title')
      
      > ax.set_ylabel('Vertical Axis Label')
       
      > ax.set_xlabel('Horizontal Axis Label')
> 
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
> Given a _pandas_ series, **series_data**, which of the following will create a histogram of series_data and align the bin edges with the horizontal tick marks?
> 
> 1 / 1 point
> 
> 
> count, bin_edges = np.histogram(series_data)
> 
> series_data.plot(type='hist', xticks = bin_edges)
> 
> 
>  series_data.plot(kind='hist')
> 
> 
> count, bin_edges = np.histogram(series_data)
> 
> series_data.plot(kind='hist', xticks = count)
> 
> 
> 

      > count, bin_edges = np.histogram(series_data)

      > series_data.plot(kind='hist', xticks = bin_edges)
> 
> 
> count, bin_edges = np.histogram(series_data)
> 
> series_data.plot(kind='hist', xticks = count, bin_edges)
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
> The following code will create a horizontal bar chart of the data in a _pandas_ dataframe, **question.**
> 
> <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> 1
> 
> question.plot(kind='barh')
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
> -- https://www.coursera.org/learn/python-for-data-visualization/exam/ryVsj/basic-visualization-tools/attempt?redirectToCover=true#Tunnel Vision Close
