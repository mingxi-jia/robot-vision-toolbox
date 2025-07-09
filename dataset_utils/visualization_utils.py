import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_voxel(np_voxels):

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    indices = np.argwhere(np_voxels[0] != 0)
    colors = np_voxels[1:, indices[:, 0], indices[:, 1], indices[:, 2]].T

    ax.scatter(indices[:, 0], indices[:, 1], indices[:, 2], color=colors/255., marker='s')

    # Set labels and show the plot
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)
    plt.show(block=False)