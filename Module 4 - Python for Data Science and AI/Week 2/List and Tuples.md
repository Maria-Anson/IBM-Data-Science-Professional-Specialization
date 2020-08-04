# List and Tuples
> 
> Latest Submission Grade
> 
> 100%
> 
>  1.Question 1
> 
> What is the syntax to obtain the first element of the tuple:
> 
> A=('a','b','c')
> 
> 1 / 1 point 
> 
>  A[1] 
> 

      A[0] 
> 
>  A[:] 
> 
> Check
> 
> Correct
> 
> correct, the index 0 corresponds to the first element of a list or tuple.
> 
>  2.Question 2
> 
> Consider the tuple **A=((1),[2,3],[4])**, that contains a tuple and list. What is the result of the following operation **A[2]**:
> 
> 1 / 1 point 
> 

      [4] 
> 
>  1 
> 
>  [2,3] 
> 
> Check
> 
> Correct
> 
> correct, the index 2 corresponds to the third element in the tuple, which contains another list.
> 
>  3.Question 3
> 
> Consider the tuple **A=((11,12),[21,22])**, that contains a tuple and list. What is the result of the following operation **A[0][1]**:
> 
> 1 / 1 point 
> 
>  21 
> 
>  11 
> 

      12 
> 
> Check
> 
> Correct
> 
> correct, A[0] corresponds to the first nested tuple; we then access the second element of the tuple using the index 1 i.e A[0][1].
> 
>  4.Question 4
> 
> What is the result of the following operation: **'1,2,3,4'.split(',')**
> 
> 1 / 1 point 
> 
>  '1','2','3','4' 
> 

      ['1','2','3','4'] 
> 
>  '1234' 
> 
>  ('1','2','3','4') 
> 
> Check
> 
> Correct
> 
> correct, split returns a **list** of the words in the string, separated by the delimiter string, in this case, a comma.
> 
>  5.Question 5
> 
> The method append does the following:
> 
> 1 / 1 point 
> 

      adds one element to a list 
> 
>  merges two lists or insert multiple elements to a list 
> 
> Check
> 
> Correct
> 
> correct, append-only adds one element.
> 
>  6.Question 6
> 
> What is an important difference between lists and tuples?
> 
> 1 / 1 point 
> 
>  Lists can't contain a string 
> 
>  Tuples can only have integers 
> 
>  Lists and tuples are the same 
> 

      Lists are mutable tuples are not 
> 
> Check
> 
> Correct
> 
> correct, lists are mutable tuples are not
> 
>  7.Question 7
> 
> consider the following list : **A=["hard rock",10,1.2]**
> 
> what will list **A** contain affter the following command is run: **del(A[1])**
> 
> 1 / 1 point 
> 
>  [10,1.2] 
> 

      ["hard rock",1.2] 
> 
>  ["hard rock",10] 
> 
> Check
> 
> Correct
> 
> correct , we will delete element 1
> 
>  8.Question 8
> 
> if **A** is a list what does the following syntax do: **B=A[:]**
> 
> 1 / 1 point 
> 
>  assigns list **A** to list **B** 
> 

      variable **B** references a new copy or clone of the original list **A** 
> 
> Check
> 
> Correct
> 
> correct
> 
>  9.Question 9
> 
> what is the result of the following: **len(("disco",10))**
> 
> 1 / 1 point 
> 

      2 
> 
>  6 
> 
>  5 
> 
> Check
> 
> Correct
> 
> correct, there are 2 elements in the tuple so the function **len** returns 2
>
> -- https://www.coursera.org/learn/python-for-applied-data-science-ai/exam/NaX6N/list-and-tuples/attempt#Tunnel Vision Close
