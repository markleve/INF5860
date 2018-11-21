import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  N = X.shape[0]
  C = W.shape[1]
  
  
  # For each N sample
  for i in range(N):
      
      # Calculating the score values for the i-th image
      scores = X[i,:].dot(W)
      
      # Guarding against overflow (numeric instability) by subtracting the 
      # maximum value from each element in the scores vector (the largest pre-
      # exponential value will then be 0)
      scores -= np.max(scores)
      
      
      # Using the softmax loss function: 

      correct_class = y[i]
      sum_score_exp = np.sum([np.exp(f) for f in scores])
      loss -= np.log(np.exp(scores[correct_class]) / sum_score_exp)

      
      # Calculating the gradient
      for j in range(C):
          dW[:,j] += (np.exp(scores[j]) / sum_score_exp)  * X[i,:]
          if j == correct_class:
              dW[:,j] -= X[i,:]

  # Dividing the total loss and dW by N
  loss /= N
  dW /= N
  
  # Adding regularization to the loss and dW
  loss += 0.5 * reg * np.sum(W**2)  
  dW += reg*W
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  N = X.shape[0]
  
  scores = X.dot(W)
  scores -= np.max(scores) # to avoid numerical instability
  
  # Calculates the sum of the exponentiated scores
  exp_scores = np.exp(scores)
  exp_scores_sum = np.sum(exp_scores, axis=1)
  exp_scores_sum = exp_scores_sum.reshape(-1,1)

  # Calculating loss function for all exponentiated scores
  n_exp_scores = exp_scores / exp_scores_sum
  loss_mat = - np.log(n_exp_scores)
  
  # Summing over only the score values that corresponds to the y
  index = np.arange(N) # Gives indexes of 0,1,2,3,...,(N-1)
  loss = (loss_mat[index,y].sum() / N) + 0.5 * reg * np.sum(W**2)
  
  # Calculating the gradient:
  n_exp_scores[index, y] -= 1
  dW = (np.dot(X.T, n_exp_scores) / N) +  reg * W


  return loss, dW

