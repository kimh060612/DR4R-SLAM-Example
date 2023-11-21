from utils import residual, rotate
from bal_dataset import read_bal_data
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

FILE_NAME = "./data/problem-16-22106-pre.txt.bz2"

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3 # 9 camera parameters (R, t) + 3 landmark points
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1
    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1
    return A

if __name__ == "__main__":
    camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = residual(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    res = least_squares(residual, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    
    ret_x = res.x
    ret_camera = np.reshape(ret_x[:n_cameras * 9], (-1, 9))
    ret_landmark = np.reshape(ret_x[n_cameras * 9 : n_points * 3], (-1, 3))
    pcd_ret = o3d.geometry.PointCloud()
    pcd_ret.points = o3d.utility.Vector3dVector(ret_landmark.astype(np.float64))
    o3d.io.write_point_cloud("./data/bal_data_ba.ply", pcd_ret)
    
    pcd_ori = o3d.geometry.PointCloud()
    pcd_ori.points = o3d.utility.Vector3dVector(points_3d)
    o3d.io.write_point_cloud("./data/bal_data_ori.ply", pcd_ori)
    
    plt.plot(res.fun)
    plt.savefig("./data/residual_result.png")
