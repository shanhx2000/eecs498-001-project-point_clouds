"""
This file stores the R and t used when we generate the pair data point cloud
"""

# ism_train_cat
theta = 0.1
R = np.array([[1,0,0],[0,np.cos(theta), -np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
t = np.array([[1, 2, -1]])

# ism_train_horse
theta = 0.2
R = np.array([[1,0,0],[0,np.cos(theta), -np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
t = np.array([[-10, 20, -30]])

# ism_train_wolf
theta = 0.3
R = np.array([[1,0,0],[0,np.cos(theta), -np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
t = np.array([[5, -10, 3]])