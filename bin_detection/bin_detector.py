'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2, os
from skimage.measure import label, regionprops
import pickle
from matplotlib import pyplot as plt

class BinDetector():
	def __init__(self):
		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		folder_path = os.path.dirname(os.path.abspath(__file__))
		model_params_file = os.path.join(folder_path, 'parameters.pkl')
		with open(model_params_file, 'rb') as f:
			params = pickle.load(f)
		self.theta_1, self.theta_2 = params[0], params[1]
		self.mu_1, self.mu_2 = params[2], params[3]
		self.sig2_1, self.sig2_2 = params[4], params[5]


	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Replace this with your own approach 

		# img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		X = img.reshape((img.shape[0] * img.shape[1]), img.shape[2]) # reshape the img to (m x n, 3)
		mask_img = np.zeros(img.shape[0] * img.shape[1], dtype=np.uint8)

		size = 300
		for i in range(len(X) // size):
			s, e = size * i, size * (i + 1)
			is_bin_prob = self.compute_probability(X[s: e], self.mu_1, self.sig2_1, self.theta_1)
			not_bin_prob = self.compute_probability(X[s: e], self.mu_2, self.sig2_2, self.theta_2)
			mask_img[s: e] = is_bin_prob > not_bin_prob

		mask_img = mask_img.reshape(img.shape[0], img.shape[1]) # reshape back

		# YOUR CODE BEFORE THIS LINE
		################################################################
		return mask_img

	def compute_probability(self, X, mu, sig2, theta):
		'''
        using the equation talked about in lecture 5 and slide 18 equation
    	Naive Bayes Model
        '''
		return np.log(theta) - 1/2 * np.log(np.linalg.det(sig2)) - 1/2 * np.diag((X - mu) @ np.linalg.inv(sig2) @ (X - mu).T)

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Replace this with your own approach

		kernel = np.ones((9, 9), np.uint8)
		erode = cv2.erode(255 * img, kernel, iterations = 1)
		dilation = cv2.dilate(erode, kernel, iterations = 1)
		_, binary = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY) # transform the image to binary
		contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find contours

		boxes = []
		for cnt in contours:
			x, y, w, h = cv2.boundingRect(cnt)
			area_ratio = cv2.contourArea(cnt) / (img.shape[0] * img.shape[1])
			if 1 <= h / w <= 2 and area_ratio > 0.005:
				boxes.append([x, y, x + w, y + h])
		
		# YOUR CODE BEFORE THIS LINE
		################################################################
		
		return boxes


if __name__ == '__main__':
	folder = 'data/validation'
	n = len(next(os.walk(folder))[2])
	for filename in os.listdir(folder):
		if filename.endswith('.jpg'):
			img = cv2.imread(os.path.join(folder, filename))
		else:
			continue
		bin_detector = BinDetector()
		mask = bin_detector.segment_image(img)
		filename = filename.split('.')[0]
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.imshow(~mask, cmap = plt.cm.binary)
		ax.axis('off')
		plt.savefig(f'{filename}_binary.jpg', format='jpg')
		plt.show()

		boxes = bin_detector.get_bounding_boxes(mask)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		for box in boxes:
			x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
			cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 9)
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.imshow(img)
		ax.axis('off')
		plt.savefig(f'{filename}_img.jpg', format='jpg')
		plt.show()
