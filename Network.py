import math
import numpy as np

def sigmoid(num):
    return 1 / (1 + math.exp(- num))

t = 0.5
w = 0.8
r = 0.1
i = 1.5


"""
init_nn takes in a list representing the shape and size of the nn. 
the first number is the number of input and the last number represents the number of 
output. Anything in between represents the number of hidden neurons 
for example, [2, 3, 1] would represents nn with 3 layers consisting of 2 input, 1 hidden layer 
with 3 neurons and an output layer consisting of 1 neuron  

Returns a numpy array containing ndarray that contains weights and biases. Each ndarray's rows are the neurons that the column neurons are connected
to. Bias is the column at the end of the ndarray.
Weights and bias are intialized to randomly between 0 and 1.
"""


def init_weights_bias(shape):
    weights = []
    num_layers = len(shape)

    for i in range(num_layers - 1):
        # +1 is for the bias
        weights.append(np.random.rand(shape[i], shape[i + 1] + 1))

    return weights

#make sure to add 1 at the end of inputs array when intializing so that np.dot would work

#takes a list of np.ndarray containing the weights and a np.array containing the inputs
#returns a list of np.ndarray. Each ndarray's rows are the sum, act and the columns are the neurons (bias at the end
#which has value sum = 0 and act = 1
def feed_forward(weights_mtrx, inputs):

    neuron_layers = []
    #the second layer of neural network.
    second_layer_sums = []
    second_layer_acts = []
    #initial value
    for i in range(len(weights_mtrx[0])):
        slayer_sum = weights_mtrx[0][i].dot(inputs)
        slayer_act = sigmoid(slayer_sum)
        second_layer_sums.append(slayer_sum)
        second_layer_acts.append(slayer_act)
    second_layer_sums.append(0)
    second_layer_acts.append(1)
    neuron_layers.append(np.array([second_layer_sums, second_layer_acts]))

    for layer in range(1, len(weights_mtrx)):
        input = neuron_layers[-1][1]
        layer_sums = []
        layer_acts = []
        for n in range(len(weights_mtrx[layer])):
            sum_n = weights_mtrx[layer][n].dot(input)
            act_n = sigmoid(sum_n)
            layer_sums.append(sum_n)
            layer_acts.append(act_n)
        #bias does not have previous value so the "input" of the bias will be always 1
        layer_sums.append(0)
        layer_acts.append(1)
        neuron_layers.append(np.array([layer_sums, layer_acts]))

    return neuron_layers

#takes a list of outputs and a lit of target outputs
#returns the sum and a list of (squared errors)/2 for each output
def squared_error_sum(output, target):
    sq_errors = []
    sq_error_sum = 0
    for i in range(len(output)):
        sq_error = 0.5 * (output[i] - target[i])**2
        sq_errors.append(sq_error)
        sq_error_sum += sq_error

    return sq_error_sum

def output_layer_back(weight_mtrx, netout_mtrx, targets):





p_input = np.array([0.1, 0.1, 1])

prac_wb = init_weights_bias([2,2,2])
print(prac_wb)
neuron_layers = feed_forward(prac_wb, p_input)
print(neuron_layers)

print(sigmoid(0.458))
p_input = np.append(p_input, 3)



print(squared_error_sum([2, 3],[0, 4]))


""""

for k in range(2000):
    #calculate the output
    sum = i * w
    output = sigmoid(sum)
    print("output:" + str(output))
    #calculate the cost
    cost = (output - t)**2

    w = w - r * output * (1 - output) * 2 * (output - t)

    print("weight:" + str(w))

"""
