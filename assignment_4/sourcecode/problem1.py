from matplotlib import pyplot as plt
import numpy as np

# input data
x1 = 0.7
x2 = 0.5
data = np.array([x1,x2])

# define activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#define matrix operation from one layer to the next 
# (defining the sigmoid function as the only activation function)
def step(input, weights):
    return sigmoid(np.matmul(input, weights))

# Input Layer
hypothetical_weights = np.array([[1,0],[0,1]])
step(data,hypothetical_weights)

assert np.array_equal(step(data,hypothetical_weights), sigmoid(data))
output1 = step(data,hypothetical_weights)

#Weights Input Layer -> First Hidden Layer
w11 = 0.9 # initialize w11
w12 = 0.3 # initialize w12
w13 = 0.9 # nintialize w13
w21 = 0.1 # initialize w11
w22 = 0.2 # initialize w12
w23 = 0.4 # nintialize w13

w12 = np.array([[w11,w12,w13],[w21,w22,w23]])
output2 = step(output1,w12)
output2

# First Hidden Layer -> Second Hidden Layer
w11 =  0.1# initialize w11
w12 =  0.8# initialize w12
w13 =  0.4# nintialize w13

w21 =  0.5# initialize w21
w22 =  0.1# initialize w22
w23 =  0.6# nintialize w23

w31 =  0.6# initialize w31
w32 =  0.7# initialize w32
w33 =  0.3# nintialize w33

w23 = np.array([[w11,w12,w13],[w21,w22,w23],[w31,w32,w33]])
output3 = step(output2,w23)

# Second Hidden Layer -> Single Neuron
w11 = 0.5 # initialize w11
w12 = 0.7 # initialize w12
w13 = 0.3 # nintialize w13

w34 = np.array([w11,w12,w13])
output4 = step(output3, w34)
print(f"The output of the neural network is {str(output4)}")
