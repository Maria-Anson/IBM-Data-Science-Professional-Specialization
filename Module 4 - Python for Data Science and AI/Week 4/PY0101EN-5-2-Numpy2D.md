<div class="alert alert-block alert-info" style="margin-top: 20px">
    <a href="https://cocl.us/topNotebooksPython101Coursera">
         <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Ad/TopAd.png" width="750" align="center">
    </a>
</div>

<a href="https://cognitiveclass.ai/">
    <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Ad/CCLog.png" width="200" align="center">
</a>

<h1>2D <code>Numpy</code> in Python</h1>

<p><strong>Welcome!</strong> This notebook will teach you about using <code>Numpy</code> in the Python Programming Language. By the end of this lab, you'll know what <code>Numpy</code> is and the <code>Numpy</code> operations.</p>

<h2>Table of Contents</h2>
<div class="alert alert-block alert-info" style="margin-top: 20px">
    <ul>
        <li><a href="create">Create a 2D Numpy Array</a></li>
        <li><a href="access">Accessing different elements of a Numpy Array</a></li>
        <li><a href="op">Basic Operations</a></li>
    </ul>
    <p>
        Estimated time needed: <strong>20 min</strong>
    </p>
</div>

<hr>

<h2 id="create">Create a 2D Numpy Array</h2>


```python
# Import the libraries

import numpy as np 
import matplotlib.pyplot as plt
```

Consider the list <code>a</code>, the list contains three nested lists **each of equal size**. 


```python
# Create a list

a = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
a
```




    [[11, 12, 13], [21, 22, 23], [31, 32, 33]]



We can cast the list to a Numpy Array as follow


```python
# Convert list to Numpy Array
# Every element is the same type

A = np.array(a)
A
```




    array([[11, 12, 13],
           [21, 22, 23],
           [31, 32, 33]])



We can use the attribute <code>ndim</code> to obtain the number of axes or dimensions referred to as the rank. 


```python
# Show the numpy array dimensions

A.ndim
```




    2



Attribute <code>shape</code> returns a tuple corresponding to the size or number of each dimension.


```python
# Show the numpy array shape

A.shape
```




    (3, 3)



The total number of elements in the array is given by the attribute <code>size</code>.


```python
# Show the numpy array size

A.size
```




    9



<hr>

<h2 id="access">Accessing different elements of a Numpy Array</h2>

We can use rectangular brackets to access the different elements of the array. The correspondence between the rectangular brackets and the list and the rectangular representation is shown in the following figure for a 3x3 array:  

<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%205/Images/NumTwoEg.png" width="500" />

We can access the 2nd-row 3rd column as shown in the following figure:

<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%205/Images/NumTwoFT.png" width="400" />

 We simply use the square brackets and the indices corresponding to the element we would like:


```python
# Access the element on the second row and third column

A[1, 2]
```




    23



 We can also use the following notation to obtain the elements: 


```python
# Access the element on the second row and third column

A[1][2]
```




    23



 Consider the elements shown in the following figure 

<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%205/Images/NumTwoFF.png" width="400" />

We can access the element as follows 


```python
# Access the element on the first row and first column

A[0][0]
```




    11



We can also use slicing in numpy arrays. Consider the following figure. We would like to obtain the first two columns in the first row

<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%205/Images/NumTwoFSF.png" width="400" />

 This can be done with the following syntax 


```python
# Access the element on the first row and first and second columns

A[0][0:2]
```




    array([11, 12])



Similarly, we can obtain the first two rows of the 3rd column as follows:


```python
# Access the element on the first and second rows and third column

A[0:2, 2]
```




    array([13, 23])



Corresponding to the following figure: 

<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%205/Images/NumTwoTST.png" width="400" />

<hr>

<h2 id="op">Basic Operations</h2>

We can also add arrays. The process is identical to matrix addition. Matrix addition of <code>X</code> and <code>Y</code> is shown in the following figure:

<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%205/Images/NumTwoAdd.png" width="500" />

The numpy array is given by <code>X</code> and <code>Y</code>


```python
# Create a numpy array X

X = np.array([[1, 0], [0, 1]]) 
X
```




    array([[1, 0],
           [0, 1]])




```python
# Create a numpy array Y

Y = np.array([[2, 1], [1, 2]]) 
Y
```




    array([[2, 1],
           [1, 2]])



 We can add the numpy arrays as follows.


```python
# Add X and Y

Z = X + Y
Z
```




    array([[3, 1],
           [1, 3]])



Multiplying a numpy array by a scaler is identical to multiplying a matrix by a scaler. If we multiply the matrix <code>Y</code> by the scaler 2, we simply multiply every element in the matrix by 2 as shown in the figure.

<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%205/Images/NumTwoDb.png" width="500" />

We can perform the same operation in numpy as follows 


```python
# Create a numpy array Y

Y = np.array([[2, 1], [1, 2]]) 
Y
```




    array([[2, 1],
           [1, 2]])




```python
# Multiply Y with 2

Z = 2 * Y
Z
```




    array([[4, 2],
           [2, 4]])



Multiplication of two arrays corresponds to an element-wise product or Hadamard product. Consider matrix <code>X</code> and <code>Y</code>. The Hadamard product corresponds to multiplying each of the elements in the same position, i.e. multiplying elements contained in the same color boxes together. The result is a new matrix that is the same size as matrix <code>Y</code> or <code>X</code>, as shown in the following figure.

<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%205/Images/NumTwoMul.png" width="500" />

We can perform element-wise product of the array <code>X</code> and <code>Y</code> as follows:


```python
# Create a numpy array Y

Y = np.array([[2, 1], [1, 2]]) 
Y
```




    array([[2, 1],
           [1, 2]])




```python
# Create a numpy array X

X = np.array([[1, 0], [0, 1]]) 
X
```




    array([[1, 0],
           [0, 1]])




```python
# Multiply X with Y

Z = X * Y
Z
```




    array([[2, 0],
           [0, 2]])



We can also perform matrix multiplication with the numpy arrays <code>A</code> and <code>B</code> as follows:

First, we define matrix <code>A</code> and <code>B</code>:


```python
# Create a matrix A

A = np.array([[0, 1, 1], [1, 0, 1]])
A
```




    array([[0, 1, 1],
           [1, 0, 1]])




```python
# Create a matrix B

B = np.array([[1, 1], [1, 1], [-1, 1]])
B
```




    array([[ 1,  1],
           [ 1,  1],
           [-1,  1]])



We use the numpy function <code>dot</code> to multiply the arrays together.


```python
# Calculate the dot product

Z = np.dot(A,B)
Z
```




    array([[0, 2],
           [0, 2]])




```python
# Calculate the sine of Z

np.sin(Z)
```




    array([[0.        , 0.90929743],
           [0.        , 0.90929743]])



We use the numpy attribute <code>T</code> to calculate the transposed matrix


```python
# Create a matrix C

C = np.array([[1,1],[2,2],[3,3]])
C
```




    array([[1, 1],
           [2, 2],
           [3, 3]])




```python
# Get the transposed of C

C.T
```




    array([[1, 2, 3],
           [1, 2, 3]])



<hr>
<h2>The last exercise!</h2>
<p>Congratulations, you have completed your first lesson and hands-on lab in Python. However, there is one more thing you need to do. The Data Science community encourages sharing work. The best way to share and showcase your work is to share it on GitHub. By sharing your notebook on GitHub you are not only building your reputation with fellow data scientists, but you can also show it off when applying for a job. Even though this was your first piece of work, it is never too early to start building good habits. So, please read and follow <a href="https://cognitiveclass.ai/blog/data-scientists-stand-out-by-sharing-your-notebooks/" target="_blank">this article</a> to learn how to share your work.
<hr>

<div class="alert alert-block alert-info" style="margin-top: 20px">
<h2>Get IBM Watson Studio free of charge!</h2>
    <p><a href="https://cocl.us/bottemNotebooksPython101Coursera"><img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Ad/BottomAd.png" width="750" align="center"></a></p>
</div>

<h3>About the Authors:</h3>  
<p><a href="https://www.linkedin.com/in/joseph-s-50398b136/" target="_blank">Joseph Santarcangelo</a> is a Data Scientist at IBM, and holds a PhD in Electrical Engineering. His research focused on using Machine Learning, Signal Processing, and Computer Vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.</p>

Other contributors: <a href="www.linkedin.com/in/jiahui-mavis-zhou-a4537814a">Mavis Zhou</a>

<hr>

<p>Copyright &copy; 2018 IBM Developer Skills Network. This notebook and its source code are released under the terms of the <a href="https://cognitiveclass.ai/mit-license/">MIT License</a>.</p>
