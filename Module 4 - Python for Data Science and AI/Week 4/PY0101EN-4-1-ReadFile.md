<div class="alert alert-block alert-info" style="margin-top: 20px">
    <a href="https://cocl.us/topNotebooksPython101Coursera">
         <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Ad/TopAd.png" width="750" align="center">
    </a>
</div>

<a href="https://cognitiveclass.ai/">
    <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Ad/CCLog.png" width="200" align="center">
</a>

<h1>Reading Files Python</h1>

<p><strong>Welcome!</strong> This notebook will teach you about reading the text file in the Python Programming Language. By the end of this lab, you'll know how to read text files.</p>

<h2>Table of Contents</h2>
<div class="alert alert-block alert-info" style="margin-top: 20px">
    <ul>
        <li><a href="download">Download Data</a></li>
        <li><a href="read">Reading Text Files</a></li>
        <li><a href="better">A Better Way to Open a File</a></li>
    </ul>
    <p>
        Estimated time needed: <strong>40 min</strong>
    </p>
</div>

<hr>

<h2 id="download">Download Data</h2>


```python
# Download Example file

!wget -O /resources/data/Example1.txt https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/labs/example1.txt
```

    --2020-06-20 19:47:17--  https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/labs/example1.txt
    Resolving s3-api.us-geo.objectstorage.softlayer.net (s3-api.us-geo.objectstorage.softlayer.net)... 67.228.254.196
    Connecting to s3-api.us-geo.objectstorage.softlayer.net (s3-api.us-geo.objectstorage.softlayer.net)|67.228.254.196|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 45 [text/plain]
    Saving to: ‘/resources/data/Example1.txt’
    
    /resources/data/Exa 100%[===================>]      45  --.-KB/s    in 0s      
    
    2020-06-20 19:47:17 (26.4 MB/s) - ‘/resources/data/Example1.txt’ saved [45/45]
    


<hr>

<h2 id="read">Reading Text Files</h2>

One way to read or write a file in Python is to use the built-in <code>open</code> function. The <code>open</code> function provides a <b>File object</b> that contains the methods and attributes you need in order to read, save, and manipulate the file. In this notebook, we will only cover <b>.txt</b> files. The first parameter you need is the file path and the file name. An example is shown as follow:

<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%204/Images/ReadOpen.png" width="500" />

 The mode argument is optional and the default value is <b>r</b>. In this notebook we only cover two modes: 
<ul>
    <li><b>r</b> Read mode for reading files </li>
    <li><b>w</b> Write mode for writing files</li>
</ul>

For the next example, we will use the text file <b>Example1.txt</b>. The file is shown as follow:

<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%204/Images/ReadFile.png" width="200" />

 We read the file: 


```python
# Read the Example1.txt

example1 = "/resources/data/Example1.txt"
file1 = open(example1, "r")
```

 We can view the attributes of the file.

The name of the file:


```python
# Print the path of file

file1.name
```




    '/resources/data/Example1.txt'



 The mode the file object is in:


```python
# Print the mode of file, either 'r' or 'w'

file1.mode
```




    'r'



We can read the file and assign it to a variable :


```python
# Read the file

FileContent = file1.read()
FileContent
```




    'This is line 1 \nThis is line 2\nThis is line 3'



The <b>/n</b> means that there is a new line. 

We can print the file: 


```python
# Print the file with '\n' as a new line

print(FileContent)
```

    This is line 1 
    This is line 2
    This is line 3


The file is of type string:


```python
# Type of file content

type(FileContent)
```




    str



 We must close the file object:


```python
# Close file after finish

file1.close()
```

<hr>

<h2 id="better">A Better Way to Open a File</h2>

Using the <code>with</code> statement is better practice, it automatically closes the file even if the code encounters an exception. The code will run everything in the indent block then close the file object. 


```python
# Open file using with

with open(example1, "r") as file1:
    FileContent = file1.read()
    print(FileContent)
```

    This is line 1 
    This is line 2
    This is line 3


The file object is closed, you can verify it by running the following cell:  


```python
# Verify if the file is closed

file1.closed
```




    True



 We can see the info in the file:


```python
# See the content of file

print(FileContent)
```

    This is line 1 
    This is line 2
    This is line 3


The syntax is a little confusing as the file object is after the <code>as</code> statement. We also don’t explicitly close the file. Therefore we summarize the steps in a figure:

<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%204/Images/ReadWith.png" width="500" />

We don’t have to read the entire file, for example, we can read the first 4 characters by entering three as a parameter to the method **.read()**:



```python
# Read first four characters

with open(example1, "r") as file1:
    print(file1.read(4))
```

    This


Once the method <code>.read(4)</code> is called the first 4 characters are called. If we call the method again, the next 4 characters are called. The output for the following cell will demonstrate the process for different inputs to the method <code>read()</code>:


```python
# Read certain amount of characters

with open(example1, "r") as file1:
    print(file1.read(4))
    print(file1.read(4))
    print(file1.read(7))
    print(file1.read(15))
```

    This
     is 
    line 1 
    
    This is line 2


The process is illustrated in the below figure, and each color represents the part of the file read after the method <code>read()</code> is called:

<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%204/Images/ReadChar.png" width="500" />

 Here is an example using the same file, but instead we read 16, 5, and then 9 characters at a time: 


```python
# Read certain amount of characters

with open(example1, "r") as file1:
    print(file1.read(16))
    print(file1.read(5))
    print(file1.read(9))
```

    This is line 1 
    
    This 
    is line 2


We can also read one line of the file at a time using the method <code>readline()</code>: 


```python
# Read one line

with open(example1, "r") as file1:
    print("first line: " + file1.readline())
```

    first line: This is line 1 
    


 We can use a loop to iterate through each line: 



```python
# Iterate through the lines

with open(example1,"r") as file1:
        i = 0;
        for line in file1:
            print("Iteration", str(i), ": ", line)
            i = i + 1;
```

    Iteration 0 :  This is line 1 
    
    Iteration 1 :  This is line 2
    
    Iteration 2 :  This is line 3


We can use the method <code>readlines()</code> to save the text file to a list: 


```python
# Read all lines and save as a list

with open(example1, "r") as file1:
    FileasList = file1.readlines()
```

 Each element of the list corresponds to a line of text:


```python
# Print the first line

FileasList[0]
```




    'This is line 1 \n'




```python
# Print the second line

FileasList[1]
```




    'This is line 2\n'




```python
# Print the third line

FileasList[2]
```




    'This is line 3'



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
