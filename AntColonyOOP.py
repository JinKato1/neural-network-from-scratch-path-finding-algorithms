import numpy as np
import random
import Task1_Utils
import time as current_time

"""
A graph repsentation of the connected coordinates
returns a dictionary of dictionaries representing which coordinate is connect to which coordinates 
and sets the initial pheromone value to 0.5 and a boolean to keep track of dead ends.  
The first key is the coordinate that you want to know which coordinates are connected from 
and the second key is the coordinates that are connected to that coordinate 

"""
class AntColony:
    def __init__(self, grid, num_ants, maxIteration, rho, Q, alpha, beta):
        self.grid = grid
        self.num_ants = num_ants
        self.maxIteration = maxIteration
        self.rho = rho
        self.Q = Q
        self.alpha = alpha
        self.beta = beta
        self.num_row, self.num_col = grid.shape
        self.pheromones = init_pheromones()


    def init_pheromones(self):
        # convert coordinates to indices
        for x in range(num_row):
            for y in range(num_col):
                pheromones[x, y] = {}
                # the end coordinate will be empty
                if x == num_row - 1 and y == num_col - 1:
                    break
                if (x, y) == (num_row - 2, num_col - 1) or (x, y) == (num_row - 1, num_col - 2):
                    pheromones[x, y][endCord] = 0.1
                    continue
                if num_col > y + 1 >= 0 and (x, y + 1) != (0, 0):
                    pheromones[x, y][x, y + 1] = 0.1
                if num_row > x + 1 >= 0 and (x + 1, y) != (0, 0):
                    pheromones[x, y][x + 1, y] = 0.1
                if num_col > y - 1 >= 0 and (x, y - 1) != (0, 0):
                    pheromones[x, y][x, y - 1] = 0.1
                if num_row > x - 1 >= 0 and (x - 1, y) != (0, 0):
                    pheromones[x, y][x - 1, y] = 0.1

        return pheromones


def update_ants(num_ants):
    bestTime = 0
    bestPath = []
    ants = []

    for ant in range(num_ants):
        time, path = ant_path()
        ants.append(ant)
        if time < bestTime:
            bestTime = time
            bestPath = path


# gets the path from start to finish of an ant
def ant_path():
    currentCord = startCord
    currentPath = [startCord]
    # deadEnds is a dictionary of list with coordinate as keys and the coordinates of dead ends as values
    deadEnds = {}
    totalTime = 0

    while currentCord != endCord:
        if currentCord in deadEnds:
            deadEnd = deadEnds[currentCord]
        else:
            deadEnd = []
        next_cord = choosePath(currentCord, currentPath, deadEnd)
        # means that it was a dead end
        if next_cord == currentCord:
            if next_cord in deadEnds:
                deadEnds.pop(next_cord)
            # updating the current path
            popped = currentPath.pop()
            try:
                deadEnds[currentPath[-1]].append(popped)
            except KeyError:
                deadEnds[currentPath[-1]] = [popped]
            currentCord = currentPath[-1]
        else:
            currentCord = next_cord
            currentPath.append(next_cord)

    # calculating the total time of the path
    for coord in currentPath:
        totalTime += weight_grid[coord]

    return totalTime, currentPath


# takes  current coordinate of the ant and the path (a list of tuples of coordinates) that the ant has taken
# returns the next coordinate that the ant probabilisticly took. Returns the input coordinate if it is a deadend
def choosePath(currentCord, currentPath, deadEnds=[]):
    tau_x_etas = {}
    tau_x_eta_sum = 0
    probabilities = {}
    for nextCord in pheromones[currentCord]:
        if nextCord not in currentPath and nextCord not in deadEnds:
            pheromone = pheromones[currentCord][nextCord]
            weight = weight_grid[nextCord]
            tau_x_eta = (pheromone ** alpha) * ((Q / weight) ** beta)
            tau_x_etas[nextCord] = tau_x_eta
            tau_x_eta_sum += tau_x_eta

    for cord in tau_x_etas:
        probabilities[cord] = tau_x_etas[cord] / tau_x_eta_sum

    # if there are no value for tau times etas list that means that it is a dead end
    if not probabilities:
        return currentCord
    elif len(probabilities) == 1:  # means there is only one choice
        return list(probabilities.keys())[0]
    else:  # implementing the roulette wheel
        probs = dict(sorted(probabilities.items(), reverse=True))
        cumulSum = []
        for i in range(len(probs)):
            sum = 0
            for j in range(i, len(probs)):
                sum += list(probs.values())[j]
            cumulSum.append(sum)
        rand_num = random.uniform(0, 1)
        chosenProb = 0
        # check where the rand_num lies in the cumulSum
        for i in range(len(cumulSum)):
            if i == len(cumulSum) - 1:
                return list(probs)[i]
            if cumulSum[i + 1] <= rand_num <= cumulSum[i]:
                return list(probs)[i]

        # finding the coordinate from the chosen probability
        # for cord, prob in probabilities.items():
        #     if prob == chosenProb:
        #         return cord


def update_pheromones(pheromones, ants):
    # evaporation for all paths
    for fromCord in pheromones:
        for toCord in pheromones[fromCord]:
            pheromones[fromCord][toCord] = pheromones[fromCord][toCord] * (1 - rho)
            # add the pheromone
    # adding the pheromone
    for idx, ant in enumerate(ants):
        if idx != len(ants) - 1:
            time = ant[0]
            path = ant[1]
            for i in range(len(path)):
                if i != len(path) - 1:
                    pheromones[path[i]][path[i + 1]] = pheromones[path[i]][path[i + 1]] + (1 / time)

    return pheromones


def run_aco(grid, num_ants=2, maxIteration=15, rho=0.01, Q=1, alpha=1, beta=1):
    # weight grid
    num_row = 5
    num_col = 5
    # initializes the weight between 1 to 10 instead of 0 to 9 since one of the equation use the weight as a denominator
    # and you can't divide by 0. The added 1 will be subtracted at the end when we find the shortest path.
    weight_grid = np.random.randint(1, 10, (num_row, num_col))
    # weight_grid = Task1_Utils.test_grids(num_row, num_col, low=1, high=10)[2][0]

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
            time, path = ant_path()
            ants.append([time, path])
            if time < bestTime:
                bestTime = time
                bestPath = path

        pheromones = update_pheromones(pheromones, ants)

    # subtracting the 1 that I initially added to avoid dividing by zero
    bestTime = bestTime - 1 * len(bestPath)


# parameters
rho = 0.01
Q = 1
alpha = 1
beta = 1
num_ants = 2
maxIteration = 15
# weight grid
num_row = 8
num_col = 8
# initializes the weight between 1 to 10 instead of 0 to 9 since one of the equation use the weight as a denominator
# and you can't divide by 0. The added 1 will be subtracted at the end when we find the shortest path.
weight_grid = np.random.randint(1, 10, (num_row, num_col))
print(weight_grid)
# weight_grid = Task1_Utils.test_grids(num_row, num_col, low=1, high=10)[2][0]


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
        time, path = ant_path()
        ants.append([time, path])
        if time < bestTime:
            bestTime = time
            bestPath = path

    pheromones = update_pheromones(pheromones, ants)

# subtracting the 1 that I initially added to avoid dividing by zero
bestTime = bestTime - 1 * len(bestPath)
print(bestPath, bestTime)

"""
[[0 5 0]
 [0 4 6]]
(2, 0)
"""
# currentCord = (0, 0)
# tau_x_etas = {}
# tau_x_eta_sum = 0
# probabilities = {}
# for nextCord in pheromones[currentCord]:
#     pheromone = pheromones[currentCord][nextCord]
#     weight = weightGrid[nextCord]
#     tau_x_eta = (pheromone ** alpha) * ((Q / weight) ** beta)
#     tau_x_etas[nextCord] = tau_x_eta
#     tau_x_eta_sum += tau_x_eta
#
# for cord in tau_x_etas:
#     probabilities[cord] = tau_x_etas[cord] / tau_x_eta_sum


# the coordinates you can go from a coordinate and its pheromones


# pheromones = {
#     (0, 0): {(0, 1): 0.5, (1, 0): 0.5},
#     (0, 1): {(1, 1): 0.5},
#     (1, 0): {(1, 1): 0.1},
#     (1, 1): {}
# }

# for fromCord, toCord in pheromones.items():
#     print(fromCord)
#     print(toCord)

"""
for fromCord in pheromones:
    for toCord in pheromones[fromCord]:
        print(pheromones[fromCord][toCord])

for fromCord, toCord in pheromones.items():
    for key in toCord:
        print(toCord[key])

for iteration in range(maxIteration):
    for ant in range(numAnts):

"""

"""
Initialize the grid and pheromone matrix
Then for t = 1 to iteration_threshhold
    for k = 1 to numAnts
        for each move until end
            let ant move based on Pij^k
        Calculate Lk
        Check if Lk is the shortest

    update pheromone by formulat Tij

"""
