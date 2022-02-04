# Programming and Mathematics For Ai

## Shortest path algorithms
Finding the shortest path for a rectangular grid (of size height x width ) where each cell has a random integer value between
0 and 9, which represent the time spent on a cell. An agent starts at the upper-left corner of the grid and must reach the lower-right
corner of the grid as fast as possible.

Inside the "Shortest Path Algorithms" folder:

DFS.py  ----> Depth-first search algorithm 

DFS_Failed_Recurion_Ver.py  -----> initial attempt at attempting to implement dfs from scratch using recursion 

Dijkstra.py ----> Dijkstra's algorithm  

AntColony.py -----> Ant colony optimization algorithm 


## Multi-layer Neural Network from scratch
The Neural Network (Neural Network.py) classifies handwritten digits. MNIST dataset (mnist.pkl.gz) is used from training and testing. 
The Neural Network has

  -Sigmoid layer with forward and backward pass 

  -Softmax output layer 
  
  -Fully Prameterizable (can specificy number and types of layers, number of units)
  
  -SGD optimizer 
  
  -Backpropagation to train the network 
  
  -92% accuracy
 
## Neural Network created using Pytorch
PyTorch is a popular open source machine learning framework developed primarily by Facebook's AI Research lab. Implemented a multi-layer Neural Network that classifies handwritten digits. 
  
  -Achieved 97.12% accuracy

