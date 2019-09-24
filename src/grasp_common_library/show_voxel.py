from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_voxel(voxel, img_path = None):
    fig = pyplot.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(voxel[:, 0], voxel[:, 1], voxel[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('voxel')
   
    pyplot.show()
    if img_path is not None:
        pyplot.savefig(img_path)
    

def convert_to_sparse_voxel_grid(voxel_grid):
    sparse_voxel_grid = []
    voxel_dim = voxel_grid.shape
    for i in xrange(voxel_dim[0]):
        for j in xrange(voxel_dim[1]):
            for k in xrange(voxel_dim[2]):
                if voxel_grid[i, j, k] == 1.:
                    sparse_voxel_grid.append([i, j, k])
    return np.asarray(sparse_voxel_grid)


if __name__ == '__main__':
    voxel_grid = np.random.rand(20, 20, 20)
    voxel_grid[voxel_grid >= 0.5] = 1.
    voxel_grid[voxel_grid <= 0.5] = 0.
    sparse_voxel_grid = convert_to_sparse_voxel_grid(voxel_grid)
    print sparse_voxel_grid
    show_voxel.plot_voxel(sparse_voxel_grid)
