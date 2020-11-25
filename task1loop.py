# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:27:51 2020

@author: katoj
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:28:36 2020

@author: katoj
"""

import numpy as np 


#Make a list with random number between 0 and 9

array = np.random.randint(0, 9, (3, 3))
num_rows, num_cols = array.shape

print(array)

#check  right or bellow 
"""
-give initial coordinate 
-create lists until you reach bottom right 
"""


isBeginning = True

#list of lists 
#each list represent a level 

level1 = [(0,0)]

tree = [[(0,0)]]
#tree = [[(0, 0)], [(0, 1)], [(0, 2)], [(1, 2)]]


#takes in the coordinates of the original node and the coordinates of the
#node that needs to be added to the tree
#if the last level includes (x, y) returns creates a new level (list)
#otherwise adds to the last level (list) of the tree
def addNode(x, y, x1, y1):
    if (x, y) in tree[len(tree) - 1]:
        tree.append([(x1, y1)])
    else:
        tree[len(tree) - 1].append((x1, y1))

def checkNode(x, y):

    lastLevel = tree[len(tree) - 1]
    
    previousNodes = []    
    for level in tree:
        previousNodes.append(level[len(level) - 1])
        

    #stop the recursion 
    
    #check if it reached goal
    if x == num_rows - 1 and y == num_cols - 1:   
        sum = 0   
        #adds the number at the coordinate, which is at the end of every level
        for level in tree:
            sum += array[level[len(level) - 1]]       
        print(sum)
        
        #checkNode the last coordinate in the previous list
        secondToLastList = tree[len(tree) - 2]
        checkNode(*(secondToLastList[len(secondToLastList) - 1]))
        
    #check right
    elif (y + 1 < num_cols and y + 1 >= 0 
        and (x, y + 1) not in (previousNodes and lastLevel)):
        
        addNode(x, y, x, y + 1)
        checkNode(x, y + 1)
        
    #check bellow
    elif (x + 1 < num_rows and x + 1 >= 0 
          and (x + 1, y) not in previousNodes and lastLevel):
        addNode(x, y, x + 1, y)
        checkNode(x + 1, y)
    #check left
    elif (y - 1 < num_cols and y - 1 >= 0 
          and (x, y - 1) not in previousNodes and lastLevel):
        addNode(x, y, x, y - 1)
        checkNode(x, y - 1)
    #check above
    elif (x - 1 < num_rows and x - 1 >= 0 
          and (x - 1, y) not in previousNodes and lastLevel):
        addNode(x, y, x - 1, y)
        checkNode(x - 1, y)
    else:
        if (x, y) == (0, 0):
            return False
        elif (x, y) in tree[len(tree) - 1]:
            secondToLastList = tree[len(tree) - 2]
            checkNode(*(secondToLastList[len(secondToLastList) - 1]))
        elif (x, y) in tree[len(tree) - 2]:
            tree.pop()
            secondToLastList = tree[len(tree) - 2]
            checkNode(*(secondToLastList[len(secondToLastList) - 1]))
"""
    if you reach here it means that the node has nowhere else to go  
    if you get to the end 
        if you are in the furthest level go back
        if you are in the second to last level 
            delete the furthest level and start from the right most element
            of the second to last level
"""    
"""
    if you get to the first node and you can't go anywhere end 
"""
    
checkNode(*(0,0))

print(tree)

"""
Note:
    if the current sum is higher than the best sum...stop
"""

"""   
you check right or bellow 
if right is empty go down
if bellow is empty go right
if right and left is empty that is the end 
take the lower one 

Have a queue 

put all 3 directions without going back
when you pop you add that value until you reach bottom right 



sum = 0
queue = []

#list of visited coordinates 
visited = []
num_rows, num_cols = array.shape

queue.append((0,0))
visited.append((0,0))


while queue:
    popped_cord = queue.pop(0)
    previously_popped = popped_cord
    
    
    row = popped_cord[0]
    col = popped_cord[1]
    sum = sum + array[popped_cord[0]][popped_cord[1]]
    
    #append the right column
    if ((row), (col + 1)) not in visited:
        queue.append(((row)(row + 1)))
    #append the bottom column
    if ((row + 1), (col)) not in visited: 
        queue.append (((row + 1), (col)))
    #append the left column
    if ((row), (col - 1)) not in visited: 
        queue.append (((row), (col - 1)))
    #append the above column 
    if ((row + 1), (col)) not in visited: 
        queue.append (((row), (col - 1)))    

for i in range(num_rows):
    for j in range(num_cols):
        print(array[i][j])


#function for checking if its a valid coordinate

def isValid(x, y):
    if (x, y) is not in visited and x >= 0 and x < num_rows and y >= 0 and y < num_cols:
        
        
"""    
 
