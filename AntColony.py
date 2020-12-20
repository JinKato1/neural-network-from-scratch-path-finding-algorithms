import numpy as np
import random
import Task1_Utils
import time as current_time
import time as currentTime

"""
A graph repsentation of the connected coordinates
returns a dictionary of dictionaries representing which coordinate is connect to which coordinates 
and sets the initial pheromone value to 0.5 and a boolean to keep track of dead ends.  
The first key is the coordinate that you want to know which coordinates are connected from 
and the second key is the coordinates that are connected to that coordinate 

"""
def init_pheromones():
    #convert coordinates to indices
    for x in range(num_row):
        for y in range(num_col):
            pheromones[x,y] = {}
            #the end coordinate will be empty
            if x == num_row - 1 and y == num_col - 1:
                break
            if (x, y) == (num_row - 2, num_col - 1) or (x, y) == (num_row - 1, num_col - 2):
                pheromones[x, y][endCord] = 0.1
                continue
            if num_col > y + 1 >= 0 and (x, y + 1) != (0, 0):
                pheromones[x,y][x, y + 1] = 0.1
            if num_row > x + 1 >= 0 and (x + 1, y) != (0, 0):
                pheromones[x, y][x + 1, y] = 0.1
            if num_col > y - 1 >= 0 and (x, y - 1) != (0, 0):
                pheromones[x, y][x, y - 1] = 0.1
            if num_row > x - 1 >= 0 and (x - 1, y) != (0, 0):
                pheromones[x, y][x - 1, y] = 0.1

    return pheromones


#gets the path from start to finish of an ant
def ant_path():
    currentCord = startCord
    currentPath = [startCord]
    #deadEnds is a dictionary of list with coordinate as keys and the coordinates of dead ends as values
    deadEnds = {}
    totalTime = 0

    while currentCord != endCord:
        next_cord = choosePath(currentCord, currentPath)
        #means that it was a dead end
        if next_cord == -1:
            return -1
        currentPath.append(next_cord)
        currentCord = next_cord

    #calculating the total time of the path
    for coord in currentPath:
        totalTime += weight_grid[coord]

    return totalTime, currentPath


#takes  current coordinate of the ant and the path (a list of tuples of coordinates) that the ant has taken
#returns the next coordinate that the ant probabilisticly took. Returns the input coordinate if it is a deadend
def choosePath(currentCord, currentPath):
    tau_x_etas = {}
    tau_x_eta_sum = 0
    probabilities = {}
    for nextCord in pheromones[currentCord]:
        if nextCord not in currentPath:
            pheromone = pheromones[currentCord][nextCord]
            weight = weight_grid[nextCord]
            tau_x_eta = (pheromone ** alpha) * ((Q / weight) ** beta)
            tau_x_etas[nextCord] = tau_x_eta
            tau_x_eta_sum += tau_x_eta

    for cord in tau_x_etas:
        probabilities[cord] = tau_x_etas[cord] / tau_x_eta_sum

    # if there are no value for tau times etas list that means that it is a dead end
    if not probabilities:
        return -1
    elif len(probabilities) == 1: #means there is only one choice
        return list(probabilities.keys())[0]
    else: #implementing the roulette wheel
        probs = dict(sorted(probabilities.items(), reverse=True))
        cumulSum = []
        for i in range(len(probs)):
            sum = 0
            for j in range(i, len(probs)):
                sum += list(probs.values())[j]
            cumulSum.append(sum)
        rand_num = random.uniform(0, 1)
        chosenProb = 0
        #check where the rand_num lies in the cumulSum
        for i in range(len(cumulSum)):
            if i == len(cumulSum) - 1:
                return list(probs)[i]
            if cumulSum[i + 1] <= rand_num <= cumulSum[i]:
                return list(probs)[i]


def update_pheromones(pheromones, ants):
    # evaporation for all paths
    for fromCord in pheromones:
        for toCord in pheromones[fromCord]:
            pheromones[fromCord][toCord] = pheromones[fromCord][toCord] * (1 - rho)
            # add the pheromone
    #adding the pheromone
    for idx, ant in enumerate(ants):
        if idx != len(ants) - 1:
            time = ant[0]
            path = ant[1]
            for i in range(len(path)):
                if i != len(path) - 1:
                    pheromones[path[i]][path[i + 1]] = pheromones[path[i]][path[i + 1]] + (1 / time)

    return pheromones


#parameters
#evaporation rate
rho = 0.5
Q = 1
alpha = 3
beta = 5
num_ants = 4
maxIteration = 100
#weight grid
#initializes the weight between 1 to 10 instead of 0 to 9 since one of the equation use the weight as a denominator
#and you can't divide by 0. The added 1 will be subtracted at the end when we find the shortest path.
#weight_grid = np.random.randint(1, 10, (num_row, num_col))

for dim in range(20, 600, 10):
    print(dim)
    time0 = currentTime.time()

    weight_grid = np.random.randint(1, 10, (dim, dim))
    num_row, num_col = weight_grid.shape

    pheromones = {}
    startCord = (0, 0)
    endCord = (num_row - 1, num_col - 1)
    # initially set to max time
    bestTime = 10 * num_row * num_col
    bestPath = []

    pheromones = init_pheromones()

    for i in range(maxIteration):
        ants = []
        for ant in range(num_ants):
            time_path = ant_path()
            while time_path == -1:
                time_path = ant_path()
            time, path = time_path
            ants.append([time, path])
            if time < bestTime:
                bestTime = time
                bestPath = path

        pheromones = update_pheromones(pheromones, ants)

    # subtracting the 1 that I initially added to avoid dividing by zero
    bestTime = bestTime - 1 * len(bestPath)

    print((currentTime.time() - time0))

"""
weight_grid = np.random.randint(0, 9, (dim, dim))
num_row, num_col = weight_grid.shape

pheromones = {}
startCord = (0, 0)
endCord = (num_row - 1, num_col - 1)
#initially set to max time
bestTime = 10 * num_row * num_col
bestPath = []

pheromones = init_pheromones()

for i in range(maxIteration):
    ants = []
    for ant in range(num_ants):
        time_path = ant_path()
        while time_path == -1:
            time_path = ant_path()
        time, path = time_path
        ants.append([time, path])
        if time < bestTime:
            bestTime = time
            bestPath = path

    pheromones = update_pheromones(pheromones, ants)

#subtracting the 1 that I initially added to avoid dividing by zero
bestTime = bestTime - 1 * len(bestPath)
print(bestPath)
print(bestTime)
"""

"""
10
0.0009963512420654297
20
0.005984067916870117
30
0.04338955879211426
40
0.05687570571899414
50
2.1218719482421875
60
15.032114028930664
70
16.117313385009766
80
more than an hour 
"""