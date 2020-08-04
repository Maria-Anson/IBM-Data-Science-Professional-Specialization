# Using loc, iloc and ix
> 
> There are three ways to select data from a data frame in Pandas: _loc_, _iloc_, and _ix_. 
> 
> ## loc
> 
> _loc_ is primarily label based; when two arguments are used, you use column headers and row indexes to select the data you want. _loc_ can also take an integer as a row or column number.
> 
> Examples of _loc_ usage:
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/kENfhU-5TMeDX4VPuczH_Q_97804255342017fb6aab1827e41b6b34_DS_4.4.3.1a-Examples-of-Loc.png?expiry=1592784000000&hmac=u_pa5HYqFWQoiex9Ivrrz1Yj3a3wdxIx6VUE_U_IKF4)
> 
> _loc_ will return a _KeyError_ if the requested items are not found.
> 
> ## iloc 
> 
> _iloc_ is integer-based. You use column numbers and row numbers to get rows or columns at particular positions in the data frame. 
> 
> Examples of _iloc_ usage:
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/hwkC8SE5SCGJAvEhORghHw_4b283976ca7258ca08473840d93a7e11_DS_4.4.3.1a-Examples-of-iloc.png?expiry=1592784000000&hmac=RJmUw2DPwgmz_25F3iKRC_RoWxEn7ctLVcVag4t-uZM)
> 
> _iloc_ will return an _IndexError_ if the requested indexer is out-of-bounds.
> 
> ## ix
> 
> By default, _ix_ looks for a label. If ix doesn't find a label, it will use an integer. This means you can select data by using either column numbers and row numbers or column headers and row names using _ix_.
> 
> In Pandas version 0.20.0 and later, _ix_ is deprecated.
> 
> ## Using loc and iloc for slicing
> 
> You can also use _loc_ and _iloc_ to slice data frames and assign the values to a new data frame. 
> 
> ### Creating a new dataframe with loc slicing
> 
> You can also slice data frames and assign the values to a new data frame using the column names. The code assigns the first three rows and all columns in between to the columns named Artist and Released. The result is a new data frame Z with the corresponding values.
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/8QAiQjs-RCaAIkI7PgQmhw_5e72f5c7aa18df4d8574986fd854bf47_DS_4.4.3.1a-loc-slice-formula.png?expiry=1592784000000&hmac=iVk2A225nhfua7iL4qiqrfwQsvDl9Vs2obgRFP4-qLo)![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/H5mGwqhuS5uZhsKobgub-A_2650318090d6787be4f36c220d1e6099_DS_4.4.3.1a-loc-slice-result.png?expiry=1592784000000&hmac=f-07KbGbSX414ntZ_j2hVquQkFzTfIZXoTG7s2Tl5tI)
> 
> ### Creating a new dataframe with iloc slicing
> 
> In this example, we assign the first two rows and the first three columns to the variable Z. The result is a data frame comprised of the selected rows and columns. 
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/KYOJWXZ_Ti6DiVl2fx4ujA_683190e1a34bbe172b718df1fe08f05d_DS_4.4.3.1a-iloc-slice-and-result.png?expiry=1592784000000&hmac=cC5b8wWQp5D2RxelmewDCyp6irnT_OfKK28WmNWc27Q)
>
> -- https://www.coursera.org/learn/python-for-applied-data-science-ai/supplement/K1rjS/using-loc-iloc-and-ix#main
