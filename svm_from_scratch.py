import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
	def __init__(self,visualization=True):
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
		step_sizes = [self.max_feature_value*0.1, self.max_feature_value*0.01,self.max_feature_value*0.001]
		#we dont need to take as small of steps with b as we do for w
		b_multiple = 3
		b_range_multiple = 3
		latest_optimum = self.max_feature_value*10

		for step in step_sizes:
			w = np.array([latest_optimum,latest_optimum])
			#convex for optimization
			optimized = false
			while not optimized:
				for b in np.arange(-1*(self.max_feature_value*b_range_multiple),self.max_feature_value*b_range_multiple, step*b_multiple):
					for trasformation in transform:
						w_t = w*trasformation
						found_option = True
						#weakest link in the svm fundamentally
						#yi(xi.w+b)>=1
						for i in self.data:
							for xi in self.data[i]:
								yi = i
								if not yi*(np.dot(w_t,xi)+b) >= 1:
									found_option = False
				if found_option:
					opt_data[np.linalg.norm(w_t)] = [w_t,b]
		if w[0] < 0:
			optimized = True
			print("optimized a step")
		else:
			w=w-step
	norms = sorted([n for n in opt_dict])
	opt_choice = norms[0]
	#||w||: [w,b]
	self.w = opt_choice[0]
	self.w = opt_choice[1]
	latest_optimum = opt_choice[0][0]+step*2		

	def predict(self, features):
		#sign x.w+b
		classification = np.sign(np.dot(np.array(features),self.w)+self.b)
		if classification != 0 and self.visualization:
			self.ax.scatter(features[0],features[1], s=200, marker='*',c= self.colors[classification])
		return classification

	def visualization(self):
		[[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

		def hyperplane(x,w,b,v):
			return (-w[0]*x-b+v / w[1])

		datarange = (self.min_feature_value*0.9,self.max_feature*1.1)
		hyp_x_min = datarange[0]
		hyp_x_max = datarange[1]

		#(w.x+b)=1
		#positive support vector hyperplane
		psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
		psv2 = hyperplane(hyp_x_max,self.w,self.b,1)
		self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2])

		#(w.x+b)=-1
		#negetive support vector hyperplane
		nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
		nsv2 = hyperplane(hyp_x_max,self.w,self.b, -1)
		self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2])

		#(w.x+b)= 0
		#positive support vector hyperplane
		db1 = hyperplane(hyp_x_min, self.w, self.b, 1)
		db2 = hyperplane(hyp_x_max,self.w,self.b,1)
		self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2])

		plt.show()


#adding data to the class
data_dict = {-1:np.array([[2,4],[3,2],[5,6],[4,7],[1,8]]), 1:np.array([[5,1],[6,-2],[5,-6],[-4,4],[-1,3]]) }

svm =Support_Vector_Machine()
svm.fit(data=data_dict)
svm.visualize()