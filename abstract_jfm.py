from cvxopt import matrix, normal
import numpy as np


class AbstractJFM:
	""" Abstract base class for [J]oint [F]eature [M]aps """
	X = None # (list of matricies) data 
	y = None # (list of vectors) state sequences (if present)

	samples = -1 # (scalar) number of training data samples
	feats = -1 # (scalar) number of features (does not coincide with get_num_dims() necessarily!)

	def __init__(self, X, y=None):
		self.X = X
		self.y = y

		# assume either co.matrix or list-of-objects
		if isinstance(X, matrix):
			(self.feats, self.samples) = X.size
		else: #list
			self.samples = len(X)
			(self.feats, foo) = X[0].shape
		print('Create structured object with #{0} training examples, each consiting of #{1} features.'.format(self.samples, self.feats))

	def get_hotstart_sol(self): 
		print('Generate a random solution vector for hot start.')
		return	normal(self.get_num_dims(), 1)

	def argmax(self, sol, idx, add_loss=False, add_prior=False): raise NotImplementedError
		
	def logsumexp(self, sol, idx, add_loss=False, add_prior=False): raise NotImplementedError

	def calc_loss(self, idx, y): raise NotImplementedError

	def get_joint_feature_map(self, idx, y=[]): raise NotImplementedError

	def get_num_samples(self):
		return self.samples

	def get_num_dims(self): raise NotImplementedError

	def evaluate(self, pred): raise NotImplementedError
