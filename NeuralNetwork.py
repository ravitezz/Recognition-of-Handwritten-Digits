# Import the functions
from layers import *
from utils import *
# Set up the packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load the training data
inputs, labels = load_images_with_labels()

# View first 8 examples
fig, ax = plt.subplots(1,8)
labl = []
for i in range(8):
    ax[i].imshow(inputs[i], cmap=mpl.cm.Greys)
    ax[i].set_title(labels[i])
plt.show()


# Pre-processing the data
train=inputs.reshape(60000,784) # reshape the inputs shape from (60000,28,28) to (60000,784)
train = np.float32(train) # change the datatype to float
train /= np.max(train,axis=1).reshape(-1,1) # Normalize the data between 0 and 1

# Now we separate the inputs into training and validation
train_ = train[0:50000,:] # We use first 50000 images for training
tr_labels = labels[0:50000]
val = train[50000:60000,:] # We use the last 10000 images for validation
val_labels = labels[50000:60000]


def forward_pass(x,w1,b1,w2,b2):
    """
    Computes the forward pass of the neural network
    
    Inputs:
    x: Numpy array of shape (N,d) where N is number of samples 
    and d is the dimension of input
    w1: numpy array of shape (d,H) where H is the size of hidden layer
    w2: numpy array of shape (H,c) where c is the number of classes
    b1: numpy array of shape (H,) 
    b2: numpy array of shape (c,)
    
    Outputs:
    probs: output of shape (N,c)
    cache_out: cache values for output layer
    cache_relu: cache values for ReLu layer
    cache_ip: cache values for input layer
    """
    probs,cache_out,cache_relu,cache_ip = None,None,None,None
    hi,cache_ip = forward_step(x,w1,b1)
    ho,cache_relu = ReLu_forward(hi)
    out,cache_out = forward_step(ho,w2,b2)
    probs = softmax(out)
    
    return(probs,cache_out,cache_relu,cache_ip)
    
def backward_pass(probs,y,cache_out,cache_relu,cache_ip):
    """
    Computes the backward pass of the neural network
    
    Inputs:
    probs: output of shape (N,c)
    cache_out: cache values for output layer
    cache_relu: cache values for ReLu layer
    cache_ip: cache values for input layer
    
    Outputs:
    loss_: loss value of the forward pass
    dw2: numpy array with samw shape as w2
    db2: numpy array with same shape as b2
    dw1: numpy array with same shape as w1
    db1: numpy array with same shape as b1
    """
    loss_,d_out = loss(probs,y) 
    dw2,db2,d_ho=backward_step(d_out,cache_out,input_layer = False)
    d_hi = ReLu_backward(d_ho,cache_relu)
    dw1,db1,_ = backward_step(d_hi,cache_ip,input_layer = True)
    
    return(loss_,dw2,db2,dw1,db1)
    
# The following function will be used to predict the labels for given images, weights and biases
def predict(X_batch,parameters):
    probs,_,_,_ = forward_pass(X_batch,parameters['w1'],parameters['b1'],parameters['w2'],parameters['b2'])
    y_pred = np.argmax(probs, axis = 1)
    return (y_pred)
  
def TwoLayerNN(learning_rate,num_iters,batch_size,train,labels,X_val,y_val):
    """
    Function to train the two layered neural network to predict MNIST data.
    Inputs: 
    learning_rate: scalar value contating learning rate for training
    num_iters: number of iterations for training
    batch_size: number of sample used for training in each iteration
    train: trainig data
    labels: labels fro the training data
    X_val: inputs for validation data
    y_val: labels for validation data
    
    Output: 
    parameters: dictionary containing trained weights and biases
    loss_history: list contating loss values for each iteration during training. It will have length of num_iters
    """
    parameters = {}
    parameters['w1'] = 0.3 * np.random.randn(784, 30)
    parameters['b1'] = np.zeros(30)
    parameters['w2'] = 0.3 * np.random.randn(30, 10)
    parameters['b2'] = np.zeros(10)
    grads={}
    loss_,grads['w1'],grads['b1'],grads['w2'],grads['b2']=None,None,None,None,None
    val_size = 10000
    loss_history=[]
    for it in range(num_iters):
        idx = np.random.choice(50000, batch_size, replace=True)
        X_batch = train[idx]
        y_batch = labels[idx]
        # The following steps implement the forward and backward pass
        probs,cache_out,cache_relu,cache_ip = forward_pass(X_batch,parameters['w1'],parameters['b1'],parameters['w2'],parameters['b2'])
        loss_,grads['w2'],grads['b2'],grads['w1'],grads['b1'] = backward_pass(probs,y_batch,cache_out,cache_relu,cache_ip)
        loss_history.append(loss_)
        #Now update the weights and biases in paramaters
        
        parameters['w1'] += -learning_rate*grads['w1']
        parameters['b1'] += -learning_rate*grads['b1']
        parameters['w2'] += -learning_rate*grads['w2']
        parameters['b2'] += -learning_rate*grads['b2']
        
        train_acc = (predict(X_batch,parameters) == y_batch).mean()
        val_acc = (predict(X_val,parameters) == y_val).mean()
        
        if it % 10 == 0:
            print ('iteration '+str(it) + ' / '+ str(num_iters) +' :loss ' + str(loss_))
            print('training accuracy: '+ str(train_acc) + ' and validation accuracy: '+ str(val_acc))
    return (parameters,loss_history)    
  
parameters,loss_history = TwoLayerNN(0.1,1000,200,train_,tr_labels,val,val_labels)

# plot the loss to see how the loss varied over iterations
plt.plot(loss_history)
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()


# Loading and pre=processing test data
t_inputs, tlabels = load_images_with_labels(type_of_data = 'testing')
tinputs=t_inputs.reshape(10000,784)
tinputs = np.float32(tinputs)
tinputs /= np.max(tinputs,axis=1).reshape(-1,1)


# Calculate the accuracy on test data using trained weights and biases
pred = predict(tinputs,parameters)
test_acc = (pred == tlabels).mean()
print(test_acc)

# Predict the labels for n images for visual representation
n=5 # number of images to predict
idx = np.random.choice(10000, n, replace=False) # select n random images from test data
labl=[]
# View first n examples
fig, ax = plt.subplots(1,n)
for i,val in enumerate(idx):
    ax[i].imshow(t_inputs[val], cmap=mpl.cm.Greys)
    ax[i].set_title(tlabels[val])
    labl.append(pred[val])
plt.show()

print('Corresponding predictions of labels for each of the above images: '+ str(labl))