%matplotlib inline

from matplotlib import pyplot as plt
import numpy as np

# define activation function and its derivative
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x)*(1-sigmoid(x))

T = np.linspace(-5,5,100)
plt.plot(T,sigmoid(T),c='r')
plt.plot(T,sigmoid_p(T),c='b')

## Assignment a)
data = [0.5, 1.5]
w1 = 1.0
w2 = 1.0
learning_rate = 1#?
b = 2.0
# calculate initial output
y = data[0]*w1 + data[1]*w2 + b
z = sigmoid(y)
z_ = 1.0
#cost function
cost = (z_-z)
# derivatives of the cost function w.r.t. weights
dcost_prediction = -1
dpred_dy = sigmoid_p(y)
dy_dw1 = data[0]
dy_dw2 = data[1]
dy_db = 1
    
# applying chain rule
dcost_dw1 = dcost_prediction * dpred_dy * dy_dw1
dcost_dw2 = dcost_prediction * dpred_dy * dy_dw2
dcost_db = dcost_prediction * dpred_dy * dy_db
    
# updating weights    
w1 = w1 - learning_rate * dcost_dw1
w2 = w2 - learning_rate * dcost_dw2
b = b - learning_rate * dcost_db

#check output with new weights
y = data[0]*w1 + data[1]*w2 + b
z = sigmoid(y)
print(f"The error of the simple network a) with updated weights is:\n {str(z_-z)}")
print(f"\nThe values of the updated weights are:\n w1={str(w1)} w2={str(w2)} b={str(b)}")

##Assignment b)
data = [0.5, 1.5]
w1 = 1.0
w2 = 1.0
learning_rate = 1#?
b = 2.0
# calculate initial output
y = data[0]*w1 + data[1]*w2 + b
z = sigmoid(y)
z_ = 1.0
#cost function
cost = np.square((z_-z))/2
# derivatives of the cost function w.r.t. weights
dcost_prediction = (z_-z)
dpred_dy = sigmoid_p(y)
dy_dw1 = data[0]
dy_dw2 = data[1]
dy_db = 1
    
# applying chain rule
dcost_dw1 = dcost_prediction * dpred_dy * dy_dw1
dcost_dw2 = dcost_prediction * dpred_dy * dy_dw2
dcost_db = dcost_prediction * dpred_dy * dy_db
    
# updating weights    
w1 = w1 - learning_rate * dcost_dw1
w2 = w2 - learning_rate * dcost_dw2
b = b - learning_rate * dcost_db

#check output with new weights
y = data[0]*w1 + data[1]*w2 + b
z = sigmoid(y)
print(f"The error of the simple network b) with updated weights is:\n {np.square((z_-z))/2}")
print(f"\nThe values of the updated weights are:\n w1={str(w1)} w2={str(w2)} b={str(b)}")
