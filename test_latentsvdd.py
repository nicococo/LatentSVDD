import cvxopt as co
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from latentsvdd import LatentSVDD
from cluster_jfm import ClusterJFM
from utils import *


if __name__ == '__main__':
	NUM_CLASSES = 6

	# generate raw training data
	Dtrain1 = SimpleToyData.get_gaussian(100,dims=2,means=[0.0,5.0],vars=[0.5,0.5])
	Dtrain2 = SimpleToyData.get_gaussian(100,dims=2,means=[-4.0,1.0],vars=[0.5,0.5])
	Dtrain3 = SimpleToyData.get_gaussian(100,dims=2,means=[4.0,-4.0],vars=[0.5,0.5])
	Dtrain4 = SimpleToyData.get_gaussian(100,dims=2,means=[-6.0,-3.0],vars=[0.5,0.5])

	Dtrain = co.matrix([[Dtrain1], [Dtrain2], [Dtrain3], [Dtrain4]])
	Dtrain = co.matrix([[Dtrain.trans()]]).trans()
	Dy = co.matrix([co.matrix([0]*100), co.matrix([1]*100), co.matrix([2]*100), co.matrix([3]*100)])

	# generate structured object
	sobj = ClusterJFM(Dtrain, y=Dy , classes=NUM_CLASSES)

	# unsupervised methods
	lsvdd = LatentSVDD(sobj, 1.0/(400.0*1.00))
	(sol, latent, threshold) = lsvdd.train_dc()

	# generate test data grid
	delta = 0.2
	x = np.arange(-8.0, 8.0, delta)
	y = np.arange(-8.0, 8.0, delta)
	X, Y = np.meshgrid(x, y)    
	(sx,sy) = X.shape
	Xf = np.reshape(X,(1,sx*sy))
	Yf = np.reshape(Y,(1,sx*sy))
	Dtest = np.append(Xf,Yf,axis=0)
	print(Dtest.shape)

	# generate structured object
	predsobj = ClusterJFM(co.matrix(Dtest), NUM_CLASSES)

	# for all methods
	fig = plt.figure()
	plt.subplot(1,2,1)
	(scores,lats) = lsvdd.apply(predsobj)

	# plot scores
	Z = np.reshape(scores,(sx,sy))
	plt.contourf(X, Y, Z)
	plt.scatter(Dtrain[0,:],Dtrain[1,:],10)

	idx = 0
	for i in range(NUM_CLASSES):
		plt.plot(sol[idx],sol[idx+1],'or')
		idx += 2

	# plot latent variable
	Z = np.reshape(lats,(sx,sy))
	plt.subplot(1,2,2)
	plt.contourf(X, Y, Z)
	plt.scatter(Dtrain[0,:],Dtrain[1,:],10)

	plt.show()
	print('finished')