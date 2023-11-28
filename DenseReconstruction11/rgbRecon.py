from scipy.spatial.transform import Rotation as R
import numpy as np
import open3d as o3d
import cv2
scale = 5000.0

if __name__ == "__main__":
    pose = []
    with open("./data/pose.txt", "r") as f:
        p = f.readlines()
        for con in p:
            qt = con[:-1].split(" ")
            trans = [ float(x) for x in [ qt[0], qt[1], qt[2] ]]
            quater = [ float(x) for x in [ qt[3], qt[4], qt[5], qt[6] ]]
            r = R.from_quat(quater).as_matrix()
            t = np.array(trans).reshape(3, 1)
            tmp = np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)
            T = np.concatenate([r, t], axis=1)
            pose.append(np.concatenate([T, tmp], axis=0))
    
    cloud = o3d.geometry.PointCloud()
    for i in range(5):
        T = pose[i]
        rgb = cv2.imread(f"./data/rgb_image/{i + 1}.png")
        depth = cv2.imread(f"./data/depth_image/{i + 1}.png", cv2.IMREAD_GRAYSCALE)
        crgb = o3d.io.read_image(f"./data/rgb_image/{i + 1}.png")
        cdepth = o3d.io.read_image(f"./data/depth_image/{i + 1}.png")
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(crgb, cdepth, scale, convert_rgb_to_intensity=False)
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        pinhole_camera_intrinsic.set_intrinsics(rgb.shape[1], rgb.shape[0], 481.2, -480.0, 319.5, 239.5)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
        pcd.transform(T)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_inlier = pcd.select_by_index(ind)
        cloud += pcd_inlier
    voxel_down_pcd = cloud.voxel_down_sample(voxel_size=0.03)
    o3d.visualization.draw_geometries([voxel_down_pcd])
    