import numpy as np
#from past.builtins import xrange


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    

    dists = self.compute_distances(X)
    
    return self.predict_labels(dists, k=k)

  

  
  

  def compute_distances(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train.


    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 

    #for i in range(num_test):
    #    dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - X[i, :]), axis=1))
    
    
    # Computes the sum of the X_train values, by summing along the column. 
    # This results in a row vector.
    X_train_sum = np.sum(self.X_train**2, axis = 1)
    
    # Computes the sum of the X_test values by summing along the column, which
    # gives a row vector. The row vector is rechaped into a column vector.
    X_test_sum = np.sum(X**2, axis = 1).reshape(-1,1)
    
    # Computing the cross product with matrix multiplication, with X_train
    # transposed. The train values are along the rows and the test values 
    # along the columns
    inner_prod = X.dot(self.X_train.T)
    
    # Computes the sqrt
    dists = np.sqrt(X_train_sum + X_test_sum - 2*inner_prod)
        
    return dists


  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    closest_y = np.zeros((1,k),dtype='int32')
    for i in range(0,num_test):
 
      # Using argsort to find the indexes of the sorted distances. Argsort sorts 
      # all the distances from the smallest distance (which is the closest) 
      # to the largest distance. 
      closest_index = np.argsort(dists[i, :])
      
      # Finding the corresponding y_train values (labels) of the sorted 
      # distances
      closest_y = self.y_train[closest_index]
      
      # Finding the y_train values corresponding to the k nearest neighbours.
      # This is a list of k labels that were the closest among the training
      # values.
      closest_y = closest_y[:k]
      
      # Finding the most common label amoung the k labels in the variable
      # closest_y.
      
      y_pred[i] = np.argmax(np.bincount(closest_y))

      # Bincount counts the number of occurrences of each value in the 
      # closesy_y array. The number of bins will be 1 larger than the largest
      # value in the closest_y array, and each bin gives the number of 
      # occurrences of its index value.
      
      # Argmax returns the indices of the maximum value. As the bincount has
      # (1 + maximum value in closest_y) number of bins (values), the indice 
      # will be the actual closest_y value.
      
    return y_pred



