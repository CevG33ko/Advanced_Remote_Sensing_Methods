import torch
import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('ggplot')

torch.manual_seed(0)

plt.rcParams["figure.figsize"] = (15, 8)


# Generating y = mx + c + random noise
num_data = 1000

# True values of m and c
m_line = 3.3
c_line = 5.3

# input (Generate random data between [-5,5])
x = 10 * torch.rand(num_data) - 5

# Output (Generate data assuming y = mx + c + noise)
y_label = m_line * x + c_line + torch.randn_like(x)
y = m_line * x + c_line

# Plot the generated data points 
plt.plot(x, y_label, '.', color='g', label="Data points")
plt.plot(x, y, color='b', label='y = mx + c', linewidth=3)
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.show()

def gradient_wrt_m_and_c(inputs, labels, m, c, k):
    
    '''
    All arguments are defined in the training section of this notebook. 
    This function will be called from the training section.  
    So before completing this function go through the whole notebook.
    
    inputs (torch.tensor): input (X)
    labels (torch.tensor): label (Y)
    m (float): slope of the line
    c (float): vertical intercept of line
    k (torch.tensor, dtype=int): random index of data points
    '''
    
    # gradient w.r.t to m is g_m 
    # gradient w.r.t to c is g_c

    X = (torch.take(inputs,k).numpy())
    Y = (torch.take(labels,k).numpy())
    
    N = k.size(0)
    sum_errors = 0
    
    # wrt m
    for i in range(0, N):
        sum_errors = sum_errors + X[i] * (Y[i] - m * X[i] - c)
    g_m = -2 * sum_errors
    
    sum_errors = 0
    
    # wrt c 
    for i in range(0, N):
        sum_errors = sum_errors + Y[i] - m * X[i] - c
    g_c = -2 * sum_errors
    
    return g_m, g_c

def update_m_and_c(m, c, g_m, g_c, lr):
    '''
    All arguments are defined in the training section of this notebook. 
    This function will be called from the training section.  
    So before completing this function go through the whole notebook.
    
    g_m = gradient w.r.t to m
    g_c = gradient w.r.t to c
    '''
    
    updated_m = m - lr*g_m
    updated_c = c - lr*g_c

    return updated_m, updated_c

# Stochastic Gradient Descent with Minibatch

# input 
X = x

# output/label
Y = y_label

num_iter = 1000
batch_size = 10

# display updated values after every 10 iterations
display_count = 20
# 

lr = 0.001
m = 2
c = 1
print()
loss = []

for i in range(0, num_iter):

    # Randomly select a training data point
    k = torch.randint(0, len(Y)-1, (batch_size,))
  
    # Calculate gradient of m and c using a mini-batch
    g_m, g_c = gradient_wrt_m_and_c(X, Y, m, c, k)
    
    # update m and c parameters
    m, c = update_m_and_c(m, c, g_m, g_c, lr)
    
    # Calculate Error
    e = Y - m * X - c
    # Compute Loss Function
    current_loss = torch.sum(torch.mul(e,e))
    loss.append(current_loss)
    

    if i % display_count==0:
        #print('Iteration: {}, Loss: {}, updated m: {:.3f}, updated c: {:.3f}'.format(i, loss[i], m, c))
        y_pred = m * X + c
        # Plot the line corresponding to the learned m and c
        plt.plot(x, y_label, '.', color='g')
        plt.plot(x, y, color='b', label='Line corresponding to m={0:.2f}, c={1:.2f}'.
                 format(m_line, c_line), linewidth=3)
        plt.plot(X, y_pred, color='r', label='Line corresponding to m_learned={0:.2f}, c_learned={1:.2f}'.
                 format(m, c), linewidth=3)
        plt.title("Iteration : {}".format(i))
        plt.legend()

        plt.ylabel('y')
        plt.xlabel('x')
        plt.show()

#print('Loss of after last batch: {}'.format(loss[-1]))
print('Leaned "m" value: {}'.format( m))
print('Leaned "c" value: {}'.format( c))

# Plot loss vs m  
plt.figure
plt.plot(range(len(loss)),loss)
plt.ylabel('loss')
plt.xlabel('iterations')
plt.show()

# Calculate the predicted y values using the learned m and c
y_pred = m * X + c

# Plot the line corresponding to the learned m and c
plt.plot(x, y_label, '.', color='g', label='X and Y')
plt.plot(x, y, color='b', label='Line corresponding to m={0:.2f}, c={1:.2f}'.format(m_line, c_line), linewidth=3)
plt.plot(X, y_pred, color='r', label='Line corresponding to m_learned={0:.2f}, c_learned={1:.2f}'.format(m, c), linewidth=3)
plt.legend()

plt.ylabel('y')
plt.xlabel('x')
plt.show()
