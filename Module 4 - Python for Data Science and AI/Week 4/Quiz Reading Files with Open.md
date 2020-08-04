# Reading Files with Open
> 
> Total points 3
> 
>  1.Question 1
> 
> Consider the following text file: **Example1.txt**:
> 
> This is line 1
> 
> This is line 2
> 
> This is line 3
> 
> What is the output of the following lines of code:
> 
> <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> 
> with open("Example1.txt","r") as file1:
> 
>     FileContent=file1.read()
> 
>     print(FileContent)
> 
> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> 
> </pre>
> 
> 1 point 
> 
>  This 
> 
>  This is line 1 
> 

     This is line 1
     
     This is line 2
     
     This is line 3 
> 
>  2.Question 2
> 
> Consider the file object: **File1**. How would you print the first two lines of text.
> 
> 1 point 
> 
>  <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> 
> file1.readline(4)
> 
> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> 
> </pre> 
> 
>  <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> 

     for n in range(0,2):
     
       print(file1.readline())
> 
> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> 
> </pre> 
> 
>  3.Question 3
> 
> Consider the following line of code:
> 
> <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> 
> with open(example1,"r") as file1:
> 
> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> 
> </pre>
> 
> What mode is the file object in?
> 
> 1 point 
> 

      read 
> 
>  append 
> 
>  write
>
> -- https://www.coursera.org/learn/python-for-applied-data-science-ai/exam/3hZOt/reading-files-with-open/attempt#Tunnel Vision Close
