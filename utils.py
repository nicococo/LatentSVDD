import cvxopt as co
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

class SimpleToyData:

	@staticmethod
	def get_n_gaussians(num=100, cluster=2, dims=2):
		dmeans = co.random((cluster, dims))
		dvars = co.ones((cluster, dims))
		data = co.matrix(0.0, (dims, num))
		for c in range(cluster):
			for d in range(dims):
				data[d,:] = co.normal(1,num)*dvars[c,d] + dmeans[c,d]
		return data

	@staticmethod
	def get_gaussian(num,dims=2,means=[0,0],vars=[1,1]):
		data = co.matrix(0.0,(dims,num))
		for d in range(dims):
			data[d,:] = co.normal(1,num)*vars[d] + means[d]
		return data

