'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''
import numpy as np
import pickle, os
from glob import glob
# from generate_rgb_data import read_pixels


class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    folder_path = os.path.dirname(os.path.abspath(__file__))
    model_params_file = os.path.join(folder_path, 'parameters.pkl')
    with open(model_params_file, 'rb') as f:
      params = pickle.load(f)
    self.theta_1, self.theta_2, self.theta_3 = params[0], params[1], params[2]
    self.mu_1, self.mu_2, self.mu_3 = params[3], params[4], params[5]
    self.sig2_1, self.sig2_2, self.sig2_3 = params[6], params[7], params[8]

  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue

	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE

    y = np.empty(len(X))
    for i in range(len(X)):
      red_prob = self.compute_probability(X[i], self.mu_1, self.sig2_1, self.theta_1)
      green_prob = self.compute_probability(X[i], self.mu_2, self.sig2_2, self.theta_2)
      blue_prob = self.compute_probability(X[i], self.mu_3, self.sig2_3, self.theta_3)
      prob_list = [red_prob, green_prob, blue_prob]
      y[i] = 1 + np.argmax(prob_list)

    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y

  def compute_probability(self, X, mu, sig2, theta):
    '''
    using the equation talked about in lecture 5 and slide 18 equation
    Naive Bayes Model
    '''
    return np.log(theta) - 1/2 * np.log(np.linalg.det(sig2)) - 1/2 * (X - mu).T @ np.linalg.inv(sig2) @ (X - mu)
