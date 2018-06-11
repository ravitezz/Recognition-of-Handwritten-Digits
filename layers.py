import numpy as np

def forward_step(x,w,b):
    """
    Computes the forward pass of the neural network
    
    Inputs:
    x: Numpy array of shape (N,d) where N is number of samples 
    and d is the dimension of input
    w: numpy array of shape (d x H) where H is size of hidden layer
    b: It is the bias matrix of shape (H,)
    
    Outputs:
    out: output of shape (N x H)
    cache: values used to calculate the output (x,w,b)
    """
    out = None
    cache = (x,w,b)   
    out = x.dot(w)+b
    
    
    return (out,cache)


def backward_step(d_out,cache,input_layer = False):
    """
    Computes the backward pass of the neural network
    
    Inputs:
    d_out: calculated derivatives 
    cache: (x,w,b) values of corresponding d_out values
    input_layer: TRUE/FALSE
    
    Outptus:
    dx: gradients with respect to x
    dw: gradients with respect to w
    db: gradients with respect to b 
    """
    x,w,b = cache
    dx,dw,db = None, None, None
    
    # dx is not valid for input layer. If input_layer is true then dx will just return None
    
    if not input_layer:
        dx = d_out.dot(w.T)  
    dw = x.T.dot(d_out)
    db = np.sum(d_out, axis=0)
    
    
    return (dw,db,dx)

def ReLu_forward(x):
    """
    Computes the ReLu activation for forward pass
    
    Inputs:
    x: numpy array of any shape
    
    Outputs:
    out : should be same shape as input
    cache: values used to calculate the output (out)
    """
    cache = x
    out = None    
    out = np.maximum(0, x)
    
    return (out,cache)

def ReLu_backward(d_out,cache):
    """
    Computes the backward pass for ReLu 
    
    Inputs: 
    d_out: derivatives of any shape
    cache: has x corresponding to d_out
    
    Outputs:
    dx: gradients with respect to x
    """
    x = cache
    dx = None
    
    dx = d_out
    dx[x < 0] = 0
    
    return (dx)

def softmax(x):
    """
    Computes the softmax loss and gradient for the neural network
    
    Inputs:
    x: numpy array of shape (N,C) where N is number of samples 
    and C is number of classes
    y: vector of labels with shape (N,)
    Outputs:
    out: softmax output of shape 
    loss: loss of forward pass (scalar value)
    dx: gradient of loss with respect to x
    """
    out = None
    shift_scores = x - np.max(x, axis = 1).reshape(-1,1)
    out = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
      
    return (out)

def loss(x,y):
    """
    Computes the softmax loss and gradient for the neural network
    
    Inputs:
    x: Matrix of shape (N,C) where N is number of samples 
    and C is number of classes
    y: vector of labels with shape (N,)
    Outputs:
    loss: loss of forward pass (scalar value)
    dx: gradient of loss with respect to x
    """
    loss,de = None,None
    num_train = x.shape[0]
    loss = -np.sum(np.log(x[range(num_train), list(y)]))
    loss /= num_train 
    de= x.copy()
    de[range(num_train), list(y)] += -1 
    de /= num_train
    return (loss,de)

