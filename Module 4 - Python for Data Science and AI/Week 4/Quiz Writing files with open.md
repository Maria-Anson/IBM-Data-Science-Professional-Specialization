# Writing files with open
> 
> Total points 3
> 
>  1.Question 1
> 
> Consider the following line of code:
> 
> <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> 1
> 
> with open("Example.txt","a") as file1:
> 
> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> 
> </pre>
> 
> What mode is the file object in?
> 
> 1 point 
> 
>  read 
> 
>  write 
> 

      append 
> 
>  2.Question 2
> 
> What do the following lines of code do?
> 
> <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> with open("Example.txt","w") as writefile:
> 
>   writefile.write("This is line A\n")
> 
>   writefile.write("This is line B\n")
> 
> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> 
> </pre>
> 
> 1 point 
> 
>  Read the file "Example.txt" 
> 

      Write to the file “Example.txt” 
> 
>  Append the file "Example.txt" 
> 
>  3.Question 3
> 
> what task do the following lines of code perform:
> 
> <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> with open('Example2.txt','r') as readfile:
> 
>     with open('Example3.txt','w') as writefile:
> 
>           for line in readfile:
> 
>                 writefile.write(line)
> 
> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> 
> </pre>
> 
> 1 point 
> 
>  print out the content of Example2.txt 
> 
>  check the mode of the open function for each file object 
> 

      copy the text from Example2.txt to Example3.txt
>
> -- https://www.coursera.org/learn/python-for-applied-data-science-ai/exam/Zk9m7/writing-files-with-open/attempt#Tunnel Vision Close
