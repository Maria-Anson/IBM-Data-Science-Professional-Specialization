<a href="https://www.bigdatauniversity.com"><img src="https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width="400" align="center"></a>

<h1 align="center"><font size="5">COLLABORATIVE FILTERING</font></h1>

Recommendation systems are a collection of algorithms used to recommend items to users based on information taken from the user. These systems have become ubiquitous can be commonly seen in online stores, movies databases and job finders. In this notebook, we will explore recommendation systems based on Collaborative Filtering and implement simple version of one using Python and the Pandas library.

<h1>Table of contents</h1>

<div class="alert alert-block alert-info" style="margin-top: 20px">
    <ol>
        <li><a href="#ref1">Acquiring the Data</a></li>
        <li><a href="#ref2">Preprocessing</a></li>
        <li><a href="#ref3">Collaborative Filtering</a></li>
    </ol>
</div>
<br>
<hr>



<a id="ref1"></a>
# Acquiring the Data

To acquire and extract the data, simply run the following Bash scripts:  
Dataset acquired from [GroupLens](http://grouplens.org/datasets/movielens/). Lets download the dataset. To download the data, we will use **`!wget`** to download it from IBM Object Storage.  
__Did you know?__ When it comes to Machine Learning, you will likely be working with large datasets. As a business, where can you host your data? IBM is offering a unique opportunity for businesses, with 10 Tb of IBM Cloud Object Storage: [Sign up now for free](http://cocl.us/ML0101EN-IBM-Offer-CC)


```python
!wget -O moviedataset.zip https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip
print('unziping ...')
!unzip -o -j moviedataset.zip 
```

    --2020-05-31 09:17:15--  https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip
    Resolving s3-api.us-geo.objectstorage.softlayer.net (s3-api.us-geo.objectstorage.softlayer.net)... 67.228.254.196
    Connecting to s3-api.us-geo.objectstorage.softlayer.net (s3-api.us-geo.objectstorage.softlayer.net)|67.228.254.196|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 160301210 (153M) [application/zip]
    Saving to: â€˜moviedataset.zipâ€™
    
    moviedataset.zip    100%[===================>] 152.88M  24.4MB/s    in 7.1s    
    
    2020-05-31 09:17:22 (21.6 MB/s) - â€˜moviedataset.zipâ€™ saved [160301210/160301210]
    
    unziping ...
    Archive:  moviedataset.zip
      inflating: links.csv               
      inflating: movies.csv              
      inflating: ratings.csv             
      inflating: README.txt              
      inflating: tags.csv                


Now you're ready to start working with the data!

<hr>

<a id="ref2"></a>
# Preprocessing

First, let's get all of the imports out of the way:


```python
#Dataframe manipulation library
import pandas as pd
#Math functions, we'll only need the sqrt function so let's import only that
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

Now let's read each file into their Dataframes:


```python
#Storing the movie information into a pandas dataframe
movies_df = pd.read_csv('movies.csv')
#Storing the user information into a pandas dataframe
ratings_df = pd.read_csv('ratings.csv')
```

Let's also take a peek at how each of them are organized:


```python
#Head is a function that gets the first N rows of a dataframe. N's default is 5.
movies_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



So each movie has a unique ID, a title with its release year along with it (Which may contain unicode characters) and several different genres in the same field. Let's remove the year from the title column and place it into its own one by using the handy [extract](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.extract.html#pandas.Series.str.extract) function that Pandas has.

Let's remove the year from the __title__ column by using pandas' replace function and store in a new __year__ column.


```python
#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
```

Let's look at the result!


```python
movies_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>Adventure|Children|Fantasy</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>Comedy|Romance</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>Comedy|Drama|Romance</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>Comedy</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>



With that, let's also drop the genres column since we won't need it for this particular recommendation system.


```python
#Dropping the genres column
movies_df = movies_df.drop('genres', 1)
```

Here's the final movies dataframe:


```python
movies_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>



<br>

Next, let's look at the ratings dataframe.


```python
ratings_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>169</td>
      <td>2.5</td>
      <td>1204927694</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2471</td>
      <td>3.0</td>
      <td>1204927438</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>48516</td>
      <td>5.0</td>
      <td>1204927435</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2571</td>
      <td>3.5</td>
      <td>1436165433</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>109487</td>
      <td>4.0</td>
      <td>1436165496</td>
    </tr>
  </tbody>
</table>
</div>



Every row in the ratings dataframe has a user id associated with at least one movie, a rating and a timestamp showing when they reviewed it. We won't be needing the timestamp column, so let's drop it to save on memory.


```python
#Drop removes a specified row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)
```

Here's how the final ratings Dataframe looks like:


```python
ratings_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>169</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2471</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>48516</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2571</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>109487</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



<hr>

<a id="ref3"></a>
# Collaborative Filtering

Now, time to start our work on recommendation systems. 

The first technique we're going to take a look at is called __Collaborative Filtering__, which is also known as __User-User Filtering__. As hinted by its alternate name, this technique uses other users to recommend items to the input user. It attempts to find users that have similar preferences and opinions as the input and then recommends items that they have liked to the input. There are several methods of finding similar users (Even some making use of Machine Learning), and the one we will be using here is going to be based on the __Pearson Correlation Function__.

<img src="https://ibm.box.com/shared/static/1ql8cbwhtkmbr6nge5e706ikzm5mua5w.png" width=800px>


The process for creating a User Based recommendation system is as follows:
- Select a user with the movies the user has watched
- Based on his rating to movies, find the top X neighbours 
- Get the watched movie record of the user for each neighbour.
- Calculate a similarity score using some formula
- Recommend the items with the highest score


Let's begin by creating an input user to recommend movies to:

Notice: To add more movies, simply increase the amount of elements in the userInput. Feel free to add more in! Just be sure to write it in with capital letters and if a movie starts with a "The", like "The Matrix" then write it in like this: 'Matrix, The' .


```python
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)
inputMovies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Breakfast Club, The</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Toy Story</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jumanji</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pulp Fiction</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Akira</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>



#### Add movieId to input user
With the input complete, let's extract the input movies's ID's from the movies dataframe and add them into it.

We can achieve this by first filtering out the rows that contain the input movies' title and then merging this subset with the input dataframe. We also drop unnecessary columns for the input to save memory space.


```python
#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
inputMovies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>296</td>
      <td>Pulp Fiction</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1274</td>
      <td>Akira</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1968</td>
      <td>Breakfast Club, The</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



#### The users who has seen the same movies
Now with the movie ID's in our input, we can now get the subset of users that have watched and reviewed the movies in our input.



```python
#Filtering out users that have watched movies that the input has watched and storing it
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>4</td>
      <td>296</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>441</th>
      <td>12</td>
      <td>1968</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>479</th>
      <td>13</td>
      <td>2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>531</th>
      <td>13</td>
      <td>1274</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>681</th>
      <td>14</td>
      <td>296</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



We now group up the rows by user ID.


```python
#Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
userSubsetGroup = userSubset.groupby(['userId'])
```

lets look at one of the users, e.g. the one with userID=1130


```python
userSubsetGroup.get_group(1130)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>104167</th>
      <td>1130</td>
      <td>1</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>104168</th>
      <td>1130</td>
      <td>2</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>104214</th>
      <td>1130</td>
      <td>296</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>104363</th>
      <td>1130</td>
      <td>1274</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>104443</th>
      <td>1130</td>
      <td>1968</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>



Let's also sort these groups so the users that share the most movies in common with the input have higher priority. This provides a richer recommendation since we won't go through every single user.


```python
#Sorting it so users with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
```

Now lets look at the first user


```python
userSubsetGroup[0:3]
```




    [(75,
            userId  movieId  rating
      7507      75        1     5.0
      7508      75        2     3.5
      7540      75      296     5.0
      7633      75     1274     4.5
      7673      75     1968     5.0),
     (106,
            userId  movieId  rating
      9083     106        1     2.5
      9084     106        2     3.0
      9115     106      296     3.5
      9198     106     1274     3.0
      9238     106     1968     3.5),
     (686,
             userId  movieId  rating
      61336     686        1     4.0
      61337     686        2     3.0
      61377     686      296     4.0
      61478     686     1274     4.0
      61569     686     1968     5.0)]



#### Similarity of users to input user
Next, we are going to compare all users (not really all !!!) to our specified user and find the one that is most similar.  
we're going to find out how similar each user is to the input through the __Pearson Correlation Coefficient__. It is used to measure the strength of a linear association between two variables. The formula for finding this coefficient between sets X and Y with N values can be seen in the image below. 

Why Pearson Correlation?

Pearson correlation is invariant to scaling, i.e. multiplying all elements by a nonzero constant or adding any constant to all elements. For example, if you have two vectors X and Y,then, pearson(X, Y) == pearson(X, 2 * Y + 3). This is a pretty important property in recommendation systems because for example two users might rate two series of items totally different in terms of absolute rates, but they would be similar users (i.e. with similar ideas) with similar rates in various scales .

![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/bd1ccc2979b0fd1c1aec96e386f686ae874f9ec0 "Pearson Correlation")

The values given by the formula vary from r = -1 to r = 1, where 1 forms a direct correlation between the two entities (it means a perfect positive correlation) and -1 forms a perfect negative correlation. 

In our case, a 1 means that the two users have similar tastes while a -1 means the opposite.

We will select a subset of users to iterate through. This limit is imposed because we don't want to waste too much time going through every single user.


```python
userSubsetGroup = userSubsetGroup[0:100]
```

Now, we calculate the Pearson Correlation between input user and subset group, and store it in a dictionary, where the key is the user Id and the value is the coefficient



```python
#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

#For every user group in our subset
for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    #Get the N for the formula
    nRatings = len(group)
    #Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    #Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0

```


```python
pearsonCorrelationDict.items()
```




    dict_items([(75, 0.8272781516947562), (106, 0.5860090386731182), (686, 0.8320502943378437), (815, 0.5765566601970551), (1040, 0.9434563530497265), (1130, 0.2891574659831201), (1502, 0.8770580193070299), (1599, 0.4385290096535153), (1625, 0.716114874039432), (1950, 0.179028718509858), (2065, 0.4385290096535153), (2128, 0.5860090386731196), (2432, 0.1386750490563073), (2791, 0.8770580193070299), (2839, 0.8204126541423674), (2948, -0.11720180773462392), (3025, 0.45124262819713973), (3040, 0.89514359254929), (3186, 0.6784622064861935), (3271, 0.26989594817970664), (3429, 0.0), (3734, -0.15041420939904673), (4099, 0.05860090386731196), (4208, 0.29417420270727607), (4282, -0.4385290096535115), (4292, 0.6564386345361464), (4415, -0.11183835382312353), (4586, -0.9024852563942795), (4725, -0.08006407690254357), (4818, 0.4885967564883424), (5104, 0.7674257668936507), (5165, -0.4385290096535153), (5547, 0.17200522903844556), (6082, -0.04728779924109591), (6207, 0.9615384615384616), (6366, 0.6577935144802716), (6482, 0.0), (6530, -0.3516054232038709), (7235, 0.6981407669689391), (7403, 0.11720180773462363), (7641, 0.7161148740394331), (7996, 0.626600514784504), (8008, -0.22562131409856986), (8086, 0.6933752452815365), (8245, 0.0), (8572, 0.8600261451922278), (8675, 0.5370861555295773), (9101, -0.08600261451922278), (9358, 0.692178738358485), (9663, 0.193972725041952), (9994, 0.5030272728659587), (10248, -0.24806946917841693), (10315, 0.537086155529574), (10368, 0.4688072309384945), (10607, 0.41602514716892186), (10707, 0.9615384615384616), (10863, 0.6020183016345595), (11314, 0.8204126541423654), (11399, 0.517260600111872), (11769, 0.9376144618769914), (11827, 0.4902903378454601), (12069, 0.0), (12120, 0.9292940047327363), (12211, 0.8600261451922278), (12325, 0.9616783115081544), (12916, 0.5860090386731196), (12921, 0.6611073566849309), (13053, 0.9607689228305227), (13142, 0.6016568375961863), (13260, 0.7844645405527362), (13366, 0.8951435925492911), (13768, 0.8770580193070289), (13888, 0.2508726030021272), (13923, 0.3516054232038718), (13934, 0.17200522903844556), (14529, 0.7417901772340937), (14551, 0.537086155529574), (14588, 0.21926450482675766), (14984, 0.716114874039432), (15137, 0.5860090386731196), (15157, 0.9035841064985974), (15466, 0.7205766921228921), (15670, 0.516015687115336), (15834, 0.22562131409856986), (16292, 0.6577935144802716), (16456, 0.7161148740394331), (16506, 0.5481612620668942), (17246, 0.48038446141526137), (17438, 0.7093169886164387), (17501, 0.8168748513121271), (17502, 0.8272781516947562), (17666, 0.7689238340176859), (17735, 0.7042381820123422), (17742, 0.3922322702763681), (17757, 0.64657575013984), (17854, 0.537086155529574), (17897, 0.8770580193070289), (17944, 0.2713848825944774), (18301, 0.29838119751643016), (18509, 0.1322214713369862)])




```python
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.827278</td>
      <td>75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.586009</td>
      <td>106</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.832050</td>
      <td>686</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.576557</td>
      <td>815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.943456</td>
      <td>1040</td>
    </tr>
  </tbody>
</table>
</div>



#### The top x similar users to input user
Now let's get the top 50 users that are most similar to the input.


```python
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>64</th>
      <td>0.961678</td>
      <td>12325</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.961538</td>
      <td>6207</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0.961538</td>
      <td>10707</td>
    </tr>
    <tr>
      <th>67</th>
      <td>0.960769</td>
      <td>13053</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.943456</td>
      <td>1040</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's start recommending movies to the input user.

#### Rating of selected users to all movies
We're going to do this by taking the weighted average of the ratings of the movies using the Pearson Correlation as the weight. But to do this, we first need to get the movies watched by the users in our __pearsonDF__ from the ratings dataframe and then store their correlation in a new column called _similarityIndex". This is achieved below by merging of these two tables.


```python
topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>1</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>2</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>3</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>6</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>
</div>



Now all we need to do is simply multiply the movie rating by its weight (The similarity index), then sum up the new ratings and divide it by the sum of the weights.

We can easily do this by simply multiplying two columns, then grouping up the dataframe by movieId and then dividing two columns:

It shows the idea of all similar users to candidate movies for the input user:


```python
#Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>weightedRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>1</td>
      <td>3.5</td>
      <td>3.365874</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>2</td>
      <td>1.5</td>
      <td>1.442517</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>3</td>
      <td>3.0</td>
      <td>2.885035</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>5</td>
      <td>0.5</td>
      <td>0.480839</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>6</td>
      <td>2.5</td>
      <td>2.404196</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_similarityIndex</th>
      <th>sum_weightedRating</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>38.376281</td>
      <td>140.800834</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.376281</td>
      <td>96.656745</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.253981</td>
      <td>27.254477</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.929294</td>
      <td>2.787882</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11.723262</td>
      <td>27.151751</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weighted average recommendation score</th>
      <th>movieId</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3.668955</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.518658</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.657941</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.316058</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Now let's sort it and see the top 20 movies that the algorithm recommended!


```python
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
recommendation_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weighted average recommendation score</th>
      <th>movieId</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5073</th>
      <td>5.0</td>
      <td>5073</td>
    </tr>
    <tr>
      <th>3329</th>
      <td>5.0</td>
      <td>3329</td>
    </tr>
    <tr>
      <th>2284</th>
      <td>5.0</td>
      <td>2284</td>
    </tr>
    <tr>
      <th>26801</th>
      <td>5.0</td>
      <td>26801</td>
    </tr>
    <tr>
      <th>6776</th>
      <td>5.0</td>
      <td>6776</td>
    </tr>
    <tr>
      <th>6672</th>
      <td>5.0</td>
      <td>6672</td>
    </tr>
    <tr>
      <th>3759</th>
      <td>5.0</td>
      <td>3759</td>
    </tr>
    <tr>
      <th>3769</th>
      <td>5.0</td>
      <td>3769</td>
    </tr>
    <tr>
      <th>3775</th>
      <td>5.0</td>
      <td>3775</td>
    </tr>
    <tr>
      <th>90531</th>
      <td>5.0</td>
      <td>90531</td>
    </tr>
  </tbody>
</table>
</div>




```python
movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2200</th>
      <td>2284</td>
      <td>Bandit Queen</td>
      <td>1994</td>
    </tr>
    <tr>
      <th>3243</th>
      <td>3329</td>
      <td>Year My Voice Broke, The</td>
      <td>1987</td>
    </tr>
    <tr>
      <th>3669</th>
      <td>3759</td>
      <td>Fun and Fancy Free</td>
      <td>1947</td>
    </tr>
    <tr>
      <th>3679</th>
      <td>3769</td>
      <td>Thunderbolt and Lightfoot</td>
      <td>1974</td>
    </tr>
    <tr>
      <th>3685</th>
      <td>3775</td>
      <td>Make Mine Music</td>
      <td>1946</td>
    </tr>
    <tr>
      <th>4978</th>
      <td>5073</td>
      <td>Son's Room, The (Stanza del figlio, La)</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>6563</th>
      <td>6672</td>
      <td>War Photographer</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>6667</th>
      <td>6776</td>
      <td>Lagaan: Once Upon a Time in India</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>9064</th>
      <td>26801</td>
      <td>Dragon Inn (Sun lung moon hak chan)</td>
      <td>1992</td>
    </tr>
    <tr>
      <th>18106</th>
      <td>90531</td>
      <td>Shame</td>
      <td>2011</td>
    </tr>
  </tbody>
</table>
</div>



### Advantages and Disadvantages of Collaborative Filtering

##### Advantages
* Takes other user's ratings into consideration
* Doesn't need to study or extract information from the recommended item
* Adapts to the user's interests which might change over time

##### Disadvantages
* Approximation function can be slow
* There might be a low of amount of users to approximate
* Privacy issues when trying to learn the user's preferences

<h2>Want to learn more?</h2>

IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems â€“ by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler">SPSS Modeler</a>

Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX">Watson Studio</a>

<h3>Thanks for completing this lesson!</h3>

<h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a></h4>
<p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clientsâ€™ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>

<hr>

<p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>
