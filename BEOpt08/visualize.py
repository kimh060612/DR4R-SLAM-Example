import numpy as np
import open3d as o3d

if __name__ == "__main__":
    pcd_ba_load = o3d.io.read_point_cloud("./data/bal_data_ba.ply")
    ba_load = np.asarray(pcd_ba_load.points)
    o3d.visualization.draw_geometries([pcd_ba_load])
    
    pcd_ori_load = o3d.io.read_point_cloud("./data/bal_data_ori.ply")
    ori_load = np.asarray(pcd_ori_load.points)
    o3d.visualization.draw_geometries([pcd_ori_load])