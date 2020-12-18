# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:28:36 2020

@author: katoj
"""

import numpy as np
import Task1_Utils
from time import time

def addNode(x, y, x1, y1, tree):
    if (x, y) in tree[len(tree) - 1]:
        tree.append([(x1, y1)])
    else:
        tree[len(tree) - 1].append((x1, y1))
    return tree


def checkNode(array):
    tree = [[(0, 0)]]
    num_rows, num_cols = array.shape
    lowestSum = 9 * num_rows * num_rows

    x = 0
    y = 0
    isFinished = True
    shortestPath = []

    while isFinished:

        lastLevel = tree[len(tree) - 1]

        previousNodes = []
        for level in tree:
            previousNodes.append(level[len(level) - 1])

        if x == num_rows - 1 and y == num_cols - 1:
            sum = 0

            for cord in previousNodes:
                sum += array[cord]

            if sum < lowestSum:
                lowestSum = sum
                shortestPath = previousNodes
            # checkNode the last coordinate in the previous list
            secondToLastList = tree[len(tree) - 2]
            x, y = secondToLastList[len(secondToLastList) - 1]

        # check right
        elif (y + 1 < num_cols and y + 1 >= 0
              and (x, y + 1) not in previousNodes and (x, y + 1) not in lastLevel):

            tree = addNode(x, y, x, y + 1, tree)
            y += 1

        # check bellow
        elif (x + 1 < num_rows and x + 1 >= 0
              and (x + 1, y) not in previousNodes and (x + 1, y) not in lastLevel):
            addNode(x, y, x + 1, y, tree)
            x += 1
        # check left
        elif (y - 1 < num_cols and y - 1 >= 0
              and (x, y - 1) not in previousNodes and (x, y - 1) not in lastLevel):
            addNode(x, y, x, y - 1, tree)
            y -= 1
        # check above
        elif (x - 1 < num_rows and x - 1 >= 0
              and (x - 1, y) not in previousNodes and (x - 1, y) not in lastLevel):
            addNode(x, y, x - 1, y, tree)
            x -= 1
        else:
            if (x, y) == (0, 0):
                isFinished = False
                return lowestSum, shortestPath

            elif (x, y) in tree[len(tree) - 1]:
                secondToLastList = tree[len(tree) - 2]
                # checkNode(*(secondToLastList[len(secondToLastList) - 1]))
                x, y = secondToLastList[len(secondToLastList) - 1]
            elif (x, y) in tree[len(tree) - 2]:
                tree.pop()
                secondToLastList = tree[len(tree) - 2]
                # checkNode(*(secondToLastList[len(secondToLastList) - 1]))
                x, y = secondToLastList[len(secondToLastList) - 1]

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



test_grids = Task1_Utils.test_grids(5, 5)
time0 = time()
array = np.random.randint(0, 9, (6, 6))
print("Training Time (in minutes) =", (time() - time0)/60)

shortest_path_cost, shortest_path = checkNode(array)

print(shortest_path_cost, shortest_path)
time0 = time()
for sample in test_grids:
    test_grid = sample[0]
    test_path = sample[1]
    #tree = [[(0, 0)]]
    shortest_path_cost, shortest_path = checkNode(test_grid)
    if test_path == shortest_path:
        print("Correct!")
print("Training Time (in minutes) =", (time() - time0)/60)
