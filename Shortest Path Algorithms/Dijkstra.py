import numpy as np
import Task1_Utils
import matplotlib
import time as time

def updateArrays(x, y, xn, yn, array, unseenNodes):
    shortestDistance = array[x][y][1]
    neighbourWeight = array[xn][yn][0]
    neighbourSD = array[xn][yn][1]
    if shortestDistance + neighbourWeight < neighbourSD:
        # updating array's shortest dist
        array[xn][yn][1] = shortestDistance + neighbourWeight
        # updating array's previous
        array[xn][yn][2] = (x, y)
        # updating the unseenNode list
        for unseenNode in unseenNodes:
            if unseenNode[1] == (xn, yn):
                unseenNode[0] = shortestDistance + neighbourWeight
    return array, unseenNodes


def run_dijkstra(weightArray):
    num_rows, num_cols = weightArray.shape
    # initially set to the absolute worst path
    max = 9 * num_rows * num_rows
    shortestDistance = []
    unseenNodes = []

    array = []

    for i in range(num_rows):
        array.append([])
        for j in range(num_cols):
            array[i].append([weightArray[i][j], max, (), "notVisited"])
    array[0][0] = [weightArray[0][0], 0, (), "visited"]

    for i in range(num_rows):
        for j in range(num_cols):
            weight, dist, prev, visitStatus = array[i][j]
            unseenNodes.append([dist, (i, j)])

    smallestDistance = max
    smallestDistanceCoord = None
    # pop this later
    smallestDistanceNodeIndex = None

    # find the coordinate with the smallest distance to check
    while unseenNodes:
        smallestDistance = max + 1
        for i in range(len(unseenNodes)):
            dist, coord = unseenNodes[i]
            if dist < smallestDistance:
                smallestDistance = dist
                smallestDistanceCoord = coord
                smallestDistanceNodeIndex = i

        x, y = smallestDistanceCoord

        if num_cols > y + 1 >= 0 and array[x][y + 1][3] == "notVisited":
            array, unseenNodes = updateArrays(x, y, x, y + 1, array, unseenNodes)
        if num_rows > x + 1 >= 0 and array[x + 1][y][3] == "notVisited":
            array, unseenNodes = updateArrays(x, y, x + 1, y, array, unseenNodes)
        if num_cols > y - 1 >= 0 and array[x][y - 1][3] == "notVisited":
            array, unseenNodes = updateArrays(x, y, x, y - 1, array, unseenNodes)
        if num_rows > x - 1 >= 0 and array[x - 1][y][3] == "notVisited":
            array, unseenNodes = updateArrays(x, y, x, y - 1, array, unseenNodes)
        # delete current node from the unseenNode list

        unseenNodes.pop(smallestDistanceNodeIndex)
        array[x][y][3] = "visited"

    currentCord = (num_rows - 1, num_cols - 1)
    shortestPath = []
    while currentCord != (0, 0):
        x, y = currentCord
        shortestPath.insert(0, currentCord)
        currentCord = array[x][y][2]

    shortestPath.insert(0, (0, 0))
    # adding the weight of the first array to the shortest distance
    shortest_path_cost = array[num_rows - 1][num_cols - 1][1] + array[0][0][0]

    return shortest_path_cost, shortestPath

weightArray = np.random.randint(0, 9, (50, 50))
run_dijkstra(weightArray)


for dim in range(50, 600, 50):
    print(dim)
    time0 = time.time()
    run_dijkstra(np.random.randint(0, 9, (dim, dim)))
    print((time.time() - time0))


#
# num_rows, num_cols = weightArray.shape
# # initially set to the absolute worst path
# max = 9 * num_rows * num_rows
# shortestDistance = []
# unseenNodes = []
#
# array = []
#
# for i in range(num_rows):
#     array.append([])
#     for j in range(num_cols):
#         array[i].append([weightArray[i][j], max, (), "notVisited"])
# array[0][0] = [weightArray[0][0], 0, (), "visited"]
#
# for i in range(num_rows):
#     for j in range(num_cols):
#         weight, dist, prev, visitStatus = array[i][j]
#         unseenNodes.append([dist, (i, j)])
#
# smallestDistance = max
# smallestDistanceCoord = None
# # pop this later
# smallestDistanceNodeIndex = None
#
# # find the coordinate with the smallest distance to check
# while unseenNodes:
#     smallestDistance = max + 1
#     for i in range(len(unseenNodes)):
#         dist, coord = unseenNodes[i]
#         if dist < smallestDistance:
#             smallestDistance = dist
#             smallestDistanceCoord = coord
#             smallestDistanceNodeIndex = i
#
#     x, y = smallestDistanceCoord
#
#     if num_cols > y + 1 >= 0 and array[x][y + 1][3] == "notVisited":
#         updateArrays(x, y, x, y + 1)
#     if num_rows > x + 1 >= 0 and array[x + 1][y][3] == "notVisited":
#         updateArrays(x, y, x + 1, y)
#     if num_cols > y - 1 >= 0 and array[x][y - 1][3] == "notVisited":
#         updateArrays(x, y, x, y - 1)
#     if num_rows > x - 1 >= 0 and array[x - 1][y][3] == "notVisited":
#         updateArrays(x, y, x, y - 1)
#     #delete current node from the unseenNode list
#
#     unseenNodes.pop(smallestDistanceNodeIndex)
#     array[x][y][3] = "visited"
#
# currentCord = (num_rows - 1, num_cols - 1)
# shortestPath = []
# while currentCord != (0, 0):
#     x, y = currentCord
#     shortestPath.insert(0, currentCord)
#     currentCord = array[x][y][2]
#
# shortestPath.insert(0, (0, 0))
# print(shortestPath)
# #adding the weight of the first array to the shortest distance
# print(array[num_rows - 1][num_cols - 1][1] + array[0][0][0])
#
#
