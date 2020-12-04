import numpy as np

"""
A graph repsentation of the connected coordinates
returns a dictionary of dictionaries representing which coordinate is connect to which coordinates 
and sets the initial pheromone value to 0.5
The first key is the coordinate that you want to know which coordinates are connected from 
and the second key is the coordinates that are connected to that coordinate 
"""
def initPheromones():
    #convert coordinates to indices
    for x in range(num_row):
        for y in range(num_col):
            pheromones[x,y] = {}
            #the end coordinate will be empty
            if x == num_row - 1 and y == num_col - 1:
                break
            if (x, y) == (num_row - 2, num_col - 1) or (x, y) == (num_row - 1, num_col - 2):
                pheromones[x, y][num_row - 1, num_row - 1] = 0.1
                continue
            if num_col > y + 1 >= 0 and (x, y + 1) != (0, 0):
                pheromones[x,y][x, y + 1] = 0.1
            if num_row > x + 1 >= 0 and (x + 1, y) != (0, 0):
                pheromones[x, y][x + 1, y] = 0.1
            if num_col > y - 1 >= 0 and (x, y - 1) != (0, 0):
                pheromones[x, y][x, y - 1] = 0.1
            if num_row > x - 1 >= 0 and (x - 1, y) != (0, 0):
                pheromones[x, y][x - 1, y] = 0.1



#gets the path from start to finish of an ant
def antPath():
    currentCord = startCord
    while currentCord != endCord:
        tau_x_etas = {}
        tau_x_eta_sum = 0
        probabilities = {}
        for nextCord in pheromones[currentCord]:
            pheromone = pheromones[currentCord][nextCord]
            weight = weightGrid[nextCord]
            tau_x_eta = (pheromone ** alpha) * ((Q / weight) ** beta)
            tau_x_etas[nextCord] = tau_x_eta
            tau_x_eta_sum += tau_x_eta

        for cord in tau_x_etas:
            probabilities[cord] = tau_x_etas[cord] / tau_x_eta_sum

        print(probabilities)

def choosePath(currentCord):
    tau_x_etas = {}
    tau_x_eta_sum = 0
    probabilities = {}
    currentPath = [startCord, (0, 1), (1, 1), (2, 0)]
    deadEnd = []

    for nextCord in pheromones[currentCord]:
        if nextCord not in currentPath:
            pheromone = pheromones[currentCord][nextCord]
            weight = weightGrid[nextCord]
            tau_x_eta = (pheromone ** alpha) * ((Q / weight) ** beta)
            tau_x_etas[nextCord] = tau_x_eta
            tau_x_eta_sum += tau_x_eta

    for cord in tau_x_etas:
        probabilities[cord] = tau_x_etas[cord] / tau_x_eta_sum

    #if there are no value for tau times etas list that means that it is a dead end
    if not tau_x_etas:
        deadEnd.append(currentCord)
    else:
        currentPath.append(currentCord)

    print(probabilities)
rho = 0.01
Q = 1
alpha = 1
beta = 1
numAnts = 4
maxIteration = 1000
num_row = 3
num_col = 3
pheromones = {}
startCord = (0, 0)
endCord = (num_row - 1, num_col - 1)
weightGrid = np.random.randint(0, 9, (num_row, num_col))


initPheromones()
choosePath((1, 0))
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

print(weightGrid)

#the coordinates you can go from a coordinate and its pheromones



# pheromones = {
#     (0, 0): {(0, 1): 0.5, (1, 0): 0.5},
#     (0, 1): {(1, 1): 0.5},
#     (1, 0): {(1, 1): 0.1},
#     (1, 1): {}
# }

# for fromCord, toCord in pheromones.items():
#     print(fromCord)
#     print(toCord)

#print(pheromones)
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
