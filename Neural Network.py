import math
import numpy as np
import gzip
import _pickle as cPickle
import random
import time as time
import pickle

def sigmoid(num):
    return 1 / (1 + math.exp(- num))


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
        layer = np.random.randn(shape[i + 1], shape[i] + 1)
        #turning the bias, which is at the end of each row to zero
        for row in range(len(layer)):
            layer[row][-1] = 0
        weights.append(layer)
    return weights

#make sure to add 1 at the end of inputs array when intializing so that np.dot would work

#takes a list of np.ndarray containing the weights and a np.array containing the inputs
#returns a list of np.ndarray. Each ndarray's rows are the sum, act and the columns are the neurons (bias at the end
#which has value sum = 0 and act = 1
def feed_forward(weights_mtrx, inputs):

    neuron_layers = []
    #the second layer of neural network.
    second_layer_acts = []
    #initial value
    for i in range(len(weights_mtrx[0])):
        slayer_sum = weights_mtrx[0][i].dot(inputs)
        slayer_act = sigmoid(slayer_sum)
        second_layer_acts.append(slayer_act)
    second_layer_acts.append(1)
    neuron_layers.append(np.array(second_layer_acts))

    for layer in range(1, len(weights_mtrx)):
        input = neuron_layers[-1]
        layer_acts = []
        #finding the the out for each neuron
        #if it is the last layer
        if layer == len(weights_mtrx) - 1:
            net_os = []
            for n in range(len(weights_mtrx[layer])):
                sum_n = weights_mtrx[layer][n].dot(input)
                net_os.append(sum_n)
            out = softmax(np.array(net_os))
            neuron_layers.append(out)
        else:
            for n in range(len(weights_mtrx[layer])):
                sum_n = weights_mtrx[layer][n].dot(input)
                act_n = sigmoid(sum_n)
                layer_acts.append(act_n)
            #bias does not have previous value so the "input" of the bias will be always 1
            layer_acts.append(1)
            neuron_layers.append(np.array([layer_acts]))

    return neuron_layers

#takes a list of outputs and a lit of target outputs
#returns the sum and a list of (squared errors)/2 for each output
def squared_error_sum(output, target):
    sq_error_sum = 0
    #for i in range(len(output[0]) - 1):
    for i in range(output[0].size):
        sq_error = 0.5 * (output[i] - target[i])**2
        sq_error_sum += sq_error

    return sq_error_sum


#j is the column of the output layre which is the row of the previous layer
#o is the row index of the output layer which indicate which output neuron it is
def output_layer_back(wb_mtrx, out_mtrx, targets, gradc_mtrx):
    out_layer = wb_mtrx[-1]
    delta_cs = []
    o_errs = []
    #calculating the error for the output neuron
    o_layer_out = out_mtrx[-1]
    for n in range(o_layer_out.size):
        target = targets[n]
        out_o = o_layer_out[n]
        o_err = -(target - out_o) * out_o * (1 - out_o)
        o_errs.append(o_err)

    #iterating through the output layer weights
    for n in range(len(out_layer)):
        for pw in range(len(out_layer[0])):
            weight = out_layer[n][pw]
            prev_out = out_mtrx[- 2][pw]
            o_err = o_errs[n]
            delta_c = o_err * prev_out
            gradc_mtrx[-1][n][pw] = gradc_mtrx[-1][n][pw] + delta_c

    return gradc_mtrx, o_errs


# def hidden_layer_back(wb_mtrx, out_mtrx, o_errs):
#
#
#
#     for o, o_err in enumerate(o_errs):
#         n_err = o_errs * wb_mtrx[-1][o][wp]
#
#     for l in range(len(wb_mtrx) - 1, 0, -1):
#
#         n_errs = []
#         current_layer = wb_mtrx[l + 1]
#         for pn in range(len(current_layer[0] - 1)):
#             n_err = 0
#             for o in range(len(o_errs)):
#                 n_err += current_layer[o][pn] * o_errs[o]
#             n_errs.append(n_err)
#
#         current_layer = wb_mtrx[l]
#         for n in range(len(current_layer)):
#             current_node = current_layer[n]
#             for pw in range(len(current_node[pw])):

def hidden_layer_back(wb_mtrx, out_mtrx, o_errs, inputs, gradc_mtrx):
    prev_errs = o_errs

    for l in range(len(wb_mtrx) - 2, -1, -1):
        n_errs = []
        prev_layer_wbs = wb_mtrx[l + 1]
        for pn in range(len(prev_layer_wbs[0]) - 1):
            n_err = 0
            for o in range(len(o_errs)):
                n_err += prev_layer_wbs[o][pn] * prev_errs[o]
            n_errs.append(n_err)
        prev_errs = n_errs

        current_layer = wb_mtrx[l]
        for n in range(len(current_layer)):
            for pn in range(len(current_layer[0])):
                if l == 0:
                    prev_out = inputs[pn]
                else:
                    prev_out = out_mtrx[l-1][pn]
                #weight = current_layer[n][pn]
                n_err = n_errs[n]
                out_h1 = out_mtrx[l][n] * (1 - out_mtrx[l][n])
                delta_c = n_err * out_h1 * prev_out
                gradc_mtrx[l][n][pn] = delta_c + gradc_mtrx[l][n][pn]

    return gradc_mtrx

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.concatenate((np.reshape(x, (784,)), [1])) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, )) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.concatenate((np.reshape(x, (784,)), [1]))for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def check_result(test_data, label, wb_mtrx):
    out = feed_forward(wb_mtrx, test_data)
    #deleting the last value in the out array since it is the activation value of bias
    output_layer = out[-1]
    #output_layer = np.delete(output_layer, [-1])
    if (np.argmax(output_layer)) == label:
        return 1
    else:
        return 0
#takes a test_datatset from mnist and the weight-bias matrix and the number of tests that will be
#executed (max 10,000 since there are 10,000 test cases in the mnist test dataset)
#returns the percentage of correct guesses
def evaluate(test_dataset, wb_mtrx, num_tests):
    if num_tests > 10000:
        raise Exception('num_tests should not be higher than 10,000')
    random.shuffle(test_dataset)
    num_correct = 0

    for i in range(num_tests):
        test_input = test_dataset[i][0]
        target = test_dataset[i][1]
        num_correct += check_result(test_input, target, wb_mtrx)

    return num_correct/num_tests


def softmax(net_os):
    try:
        out = math.e ** net_os / np.sum(math.e ** net_os)
    except ZeroDivisionError:
        print(net_os)
        return out
    return out

#Parameters: the matrix of weight & bias, the matrix of gradient of c
#Returns: a matrix with update weight & bias
def update_wb_mtrx(wb_mtrx, gradc_mtrx):
    new_wb_mtrx = []
    for i in range(len(wb_mtrx)):
        new_wb_mtrx.append(wb_mtrx[i] - ((learning_rate / (batch_size)) * gradc_mtrx[i]))
    return new_wb_mtrx

# def SGD(training_data, epochs, mini_batch_size):
#     for i in epochs:
#         random.shuffle(training_data)
#         mini_batches =

def minibatch_sgd(mini_batches, wb_mtrx, epochs):
    for epoch in range(epochs):
        time0 = time.time()
        for idx, mini_batch in enumerate(mini_batches):

            # if idx % 200 == 0:
            #     print(time.time() - time0)
            #     print(evaluate(test_data, wb_mtrx, 10000))
            gradc_mtrx = []
            for mtrx in wb_mtrx:
                gradc_mtrx.append(np.zeros(mtrx.shape))

            for sample in mini_batch:
                inputs = sample[0]
                target = sample[1].flatten()

                out_mtrx = feed_forward(wb_mtrx, inputs)
                gradc_mtrx, o_errs = output_layer_back(wb_mtrx, out_mtrx, target, gradc_mtrx)
                gradc_mtrx = hidden_layer_back(wb_mtrx, out_mtrx, o_errs, inputs, gradc_mtrx)

            wb_mtrx = update_wb_mtrx(wb_mtrx, gradc_mtrx)

            accuracy = evaluate(test_data, wb_mtrx, 10000)

        print("Training Time:" + str(time.time() - time0) + " "+str(accuracy))
    return wb_mtrx
#not working version of

#hyper parameters
learning_rate = 3
nn_shape_size = [784, 30, 10]

training_data, validation_data, test_data = load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

wb_mtrx = init_weights_bias(nn_shape_size)



#matrix used to store the gradient of C, initiazed with 0

batch_size = 100
epochs = 3

mini_batches = [training_data[i: i + batch_size] for i in range(0, len(training_data), batch_size)]



with open("wb_mtrx", "rb") as f:
    wb_mtrx = pickle.load(f)

wb_mtrx = minibatch_sgd(mini_batches, wb_mtrx, epochs)

# with open("wb_mtrx_l3-30-e4-batch30.txt", "wb") as f:
#     cPickle.dump(wb_mtrx,f)

#print(evaluate(test_data, wb_mtrx, 10000))
# wb_mtrx = init_weights_bias([784, 30, 10])
#
# print(evaluate(test_data, wb_mtrx, 1000))
# for i in range(20000):
#     inputs = training_data[i][0].flatten()
#     inputs = inputs.tolist()
#     inputs.append(1)
#
#     target = training_data[i][1].flatten()
#     target = target.tolist()
#
#     out_mtrx = feed_forward(wb_mtrx, inputs)
#     # if i % 10000 == 0:
#     #      print(squared_error_sum(out_mtrx[-1], target))
#     #print(squared_error_sum(out_mtrx[-1], target))
#     new_out_w, o_errs = output_layer_back(wb_mtrx, out_mtrx, target)
#     new_wb = hidden_layer_back(wb_mtrx, out_mtrx, o_errs, inputs)
#     new_wb.append(new_out_w)
#     wb_mtrx = new_wb
#
# print(evaluate(test_data, wb_mtrx, 1000))


# training_data, validation_data, test_data = load_data_wrapper()
# training_data = list(training_data)
#
# random.shuffle(training_data)
#
# inputs = np.concatenate((training_data[0][0].flatten(), [1]))
# target = training_data[0][1].flatten()
# wb_mtrx = init_weights_bias([784, 10, 10])
#
# for i in range(10000):
#     out_mtrx = feed_forward(wb_mtrx, inputs)
#     new_out_w, o_errs = output_layer_back(wb_mtrx, out_mtrx, target)
#     new_wb = hidden_layer_back(wb_mtrx, out_mtrx, o_errs, inputs)
#     new_wb.append(new_out_w)
#     print(squared_error_sum(out_mtrx[-1], target))
#     wb_mtrx = new_wb
# print(wb_mtrx)
# print(feed_forward(wb_mtrx, inputs)[-1])
# print(target)




# inputs = [0.05, 0.1, 0.3, 0.4, 1]
# target = [0.01, 0.99, 0.5, 0.5]
# wb_mtrx = init_weights_bias([4, 3, 4])
# print(wb_mtrx)
# print(feed_forward(wb_mtrx, inputs)[-1])
#
# for i in range(1000):
#     out_mtrx = feed_forward(wb_mtrx, inputs)
#     print(squared_error_sum(out_mtrx[-1], target))
#     new_out_w, o_errs = output_layer_back(wb_mtrx, out_mtrx, target)
#     new_wb = hidden_layer_back(wb_mtrx, out_mtrx, o_errs, inputs)
#     new_wb.append(new_out_w)
#     wb_mtrx = new_wb
# print(wb_mtrx)
# print(feed_forward(wb_mtrx, inputs)[-1])
# print(target)



#working version of the example
# wb_mtrx = [np.array([[0.15, 0.20, 0.35], [0.25, 0.3, 0.35]]),
#            np.array([[0.40, 0.45, 0.6], [0.50, 0.55, 0.6]])]
# wb_mtrx = init_weights_bias(nn_shape_size)
# ex_wb = [np.array([[0.149780716, 0.19956143, 0.35], [0.24975114, 0.29950229, 0.35]]),
#            np.array([[0.35891648, 0.408666186, 0.6], [0.511301270, 0.561370121, 0.6]])]
# ex_input = [0.05, 0.1, 1]
# ex_target = [0.01, 0.99]
"""
inputs = training_data[0][0]
target = training_data[0][1].flatten()
print("target:" + str(target))

output = feed_forward(wb_mtrx, inputs)
print(output[-1] * 100)
print(np.argmax(output[-1] * 100))
print(evaluate([(inputs,5)],wb_mtrx, 1))
for i in range(500):
    gradc_mtrx = []
    for mtrx in wb_mtrx:
        gradc_mtrx.append(np.zeros(mtrx.shape))

    out_mtrx = feed_forward(wb_mtrx, inputs)
    gradc_mtrx, o_errs = output_layer_back(wb_mtrx, out_mtrx, target, gradc_mtrx)
    gradc_mtrx = hidden_layer_back(wb_mtrx, out_mtrx, o_errs, inputs, gradc_mtrx)

    wb_mtrx = update_wb_mtrx(wb_mtrx, gradc_mtrx)

output = feed_forward(wb_mtrx, inputs)
out = output[-1]
print(np.argmax(out))
print(evaluate([(inputs,5)], wb_mtrx, 1))
"""