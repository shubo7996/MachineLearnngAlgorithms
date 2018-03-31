import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
	def __init__(self,visualization=true):
		self.visualization = true
		self.colors = {1:'r', -1:'b'}
		if self.visualization:
			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(1,1,1)
	#train the dataset
	def fit(self, data):
		self.data = data
		#{||w||: [w,b]}
		opt_dict = {}
		transform = ([1,1],[-1,1],[-1,-1],[1,-1])
		all_data = []
		for yi in self.data:
			for featureSet in self.data[yi]:
				for feature in featureSet:
					all_data.append(feature)

		self.max_feature_value = max(all_data)
		self.min_feature_value = min(all_data)

		pass

	def predict(self, features):
		#sign x.w+b
		classification = np.sign(np.dot(np.array(features),self.w)+self.b)
		return classification


#adding data to the class
data_dict = {-1:np.array([[2,4],[3,2],[5,6],[4,7],[1,8]]), 1:np.array([[5,1],[6,-2],[5,-6],[-4,4],[-1,3]]) }