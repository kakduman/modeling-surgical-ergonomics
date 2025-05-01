
import numpy as np
import open3d as o3d
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--colab_environment', type=str, required=False, help='present if in Colab Environment')

parser.add_argument('--back_intrinsics_path', type=str, required=True, help='path to the back intrinsics.npy file')
parser.add_argument('--left_intrinsics_path', type=str, required=True, help='path to the left intrinsics.npy file')
parser.add_argument('--right_intrinsics_path', type=str, required=True, help='path to the right intrinsics.npy file')

parser.add_argument('--back_image_path', type=str, required=True, help='path to the back image file')
parser.add_argument('--left_image_path', type=str, required=True, help='path to the left image file')
parser.add_argument('--right_image_path', type=str, required=True, help='path to the right image file')

args = parser.parse_args()

COLAB_FLAG = False
# if in Colab environment, do not visualize
if args.colab_environment is not None:
    COLAB_FLAG = True

# source for o3d tutorial: http://www.open3d.org/docs/latest/tutorial/Basic/rgbd_odometry.html
intrinsic_matrix_target = np.load(args.back_intrinsics_path)
intrinsic_matrix_source = np.load(args.left_intrinsics_path)
intrinsic_matrix_source_2 = np.load(args.right_intrinsics_path)

intrinsic_matrix_source = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsic_matrix_source[0][0], intrinsic_matrix_source[1][1], intrinsic_matrix_source[0][2], intrinsic_matrix_source[1][2])
intrinsic_matrix_target = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsic_matrix_target[0][0], intrinsic_matrix_target[1][1], intrinsic_matrix_target[0][2], intrinsic_matrix_target[1][2])
intrinsic_matrix_source_2 = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsic_matrix_source_2[0][0], intrinsic_matrix_source_2[1][1], intrinsic_matrix_source_2[0][2], intrinsic_matrix_source_2[1][2])

target_image = o3d.io.read_image(args.back_image_path)
target_depth = o3d.io.read_image(args.back_image_path.replace("image", "depth"))

source_image = o3d.io.read_image(args.left_image_path)
source_depth = o3d.io.read_image(args.left_image_path.replace("image", "depth"))

source_2_image = o3d.io.read_image(args.right_image_path)
source_2_depth = o3d.io.read_image(args.right_image_path.replace("image", "depth"))

target_image_depth = o3d.geometry.RGBDImage.create_from_color_and_depth(target_image, target_depth)
source_image_depth = o3d.geometry.RGBDImage.create_from_color_and_depth(source_image, source_depth)
source_2_image_depth = o3d.geometry.RGBDImage.create_from_color_and_depth(source_2_image, source_2_depth)

target_point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(target_image_depth, intrinsic_matrix_target)
source_point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(source_image_depth, intrinsic_matrix_source)
source_2_point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(source_2_image_depth, intrinsic_matrix_source_2)

# RGBD Odometry
odometry_option = o3d.pipelines.odometry.OdometryOption(iteration_number_per_pyramid_level=o3d.utility.IntVector([200,200,200]),
                                                        depth_diff_max=1.0,
                                                        depth_min=0.000000,
                                                        depth_max=40.000000)
odometry_init = np.identity(4)

# source -> target
# all camera intrinsics should be roughly the same, so just use the target one (odometry technically assumes same cam)
[success_hybrid_term, trans_hybrid_term, info] = o3d.pipelines.odometry.compute_rgbd_odometry(source_image_depth,
                                                                                              target_image_depth,
                                                                                              intrinsic_matrix_target,
                                                                                              odometry_init,
                                                                                              o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                                                                                              odometry_option)
# source2 -> target
[success_hybrid_term_2, trans_hybrid_term_2, info_2] = o3d.pipelines.odometry.compute_rgbd_odometry(source_2_image_depth,
                                                                                                    target_image_depth,
                                                                                                    intrinsic_matrix_target,
                                                                                                    odometry_init,
                                                                                                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                                                                                                    odometry_option)
# visualizations
if success_hybrid_term:
    print("Local Registration: Source to Target Transformation: ", trans_hybrid_term)

    source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(source_image_depth, intrinsic_matrix_source)

    if not COLAB_FLAG:
        o3d.visualization.draw_geometries([target_point_cloud, source_pcd_hybrid_term],
                                          zoom=0.65,
                                          front=[0, 0, -1],
                                          lookat=[0, 0, 1.5],
                                          up=[0, -1, 0])

    # now do the transformation
    source_pcd_hybrid_term.transform(trans_hybrid_term)

    if not COLAB_FLAG:
        o3d.visualization.draw_geometries([target_point_cloud, source_pcd_hybrid_term],
                                          zoom=0.65,
                                          front=[0, 0, -1],
                                          lookat=[0, 0, 1.5],
                                          up=[0, -1, 0])

if success_hybrid_term_2:
    print("Local Registration: Source_2 to Target Transformation: ", trans_hybrid_term_2)

    source_pcd_hybrid_term_2 = o3d.geometry.PointCloud.create_from_rgbd_image(source_2_image_depth, intrinsic_matrix_source_2)

    if not COLAB_FLAG:
        o3d.visualization.draw_geometries([target_point_cloud, source_pcd_hybrid_term_2],
                                          zoom=0.65,
                                          front=[0, 0, -1],
                                          lookat=[0, 0, 1.5],
                                          up=[0, -1, 0])

    # now do the transformation
    source_pcd_hybrid_term_2.transform(trans_hybrid_term_2)

    if not COLAB_FLAG:
        o3d.visualization.draw_geometries([target_point_cloud, source_pcd_hybrid_term_2],
                                          zoom=0.65,
                                          front=[0, 0, -1],
                                          lookat=[0, 0, 1.5],
                                          up=[0, -1, 0])


# Global Registration: tutorial at http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    pcd_down.estimate_normals()

    radius_feature = voxel_size * 5 * 20

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
                                                               o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh


voxel_size = 0.30

source_down, source_fpfh = preprocess_point_cloud(source_point_cloud, voxel_size)
source_2_down, source_2_fpfh = preprocess_point_cloud(source_2_point_cloud, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target_point_cloud, voxel_size)


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5 * 40

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(source_down,
                                                                                      target_down,
                                                                                      source_fpfh,
                                                                                      target_fpfh,
                                                                                      True,
                                                                                      distance_threshold,
                                                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                                                                                      5,
                                                                                      [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.5),
                                                                                       o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
                                                                                      o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=1000000, confidence=1.0))
    return result


result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
print("Global Registration: Source to Target Transformation:", result_ransac.transformation)

if not COLAB_FLAG:
    o3d.visualization.draw_geometries([target_point_cloud, source_point_cloud],
                                      zoom=0.65,
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 1.5],
                                      up=[0, -1, 0])

source_pcd_temp = copy.deepcopy(source_point_cloud)
source_pcd_temp.transform(result_ransac.transformation)

if not COLAB_FLAG:
    o3d.visualization.draw_geometries([target_point_cloud, source_pcd_temp],
                                      zoom=0.65,
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 1.5],
                                      up=[0, -1, 0])

result_ransac_2 = execute_global_registration(source_2_down, target_down,
                                              source_2_fpfh, target_fpfh,
                                              voxel_size)
print("Global Registration: Source_2 to Target Transformation:", result_ransac_2.transformation)

if not COLAB_FLAG:
    o3d.visualization.draw_geometries([target_point_cloud, source_2_point_cloud],
                                      zoom=0.65,
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 1.5],
                                      up=[0, -1, 0])

source_2_pcd_temp = copy.deepcopy(source_2_point_cloud)
source_2_pcd_temp.transform(result_ransac_2.transformation)

if not COLAB_FLAG:
    o3d.visualization.draw_geometries([target_point_cloud, source_2_pcd_temp],
                                      zoom=0.65,
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 1.5],
                                      up=[0, -1, 0])


# Using Global Registration matrix as initial odometry matrix
odometry_option = o3d.pipelines.odometry.OdometryOption(iteration_number_per_pyramid_level=o3d.utility.IntVector([200,200,200]),
                                                        depth_diff_max=1.0,
                                                        depth_min=0.000000,
                                                        depth_max=40.000000)
odometry_init = result_ransac.transformation
odometry_init_2 = result_ransac_2.transformation

# source -> target
[success_hybrid_term, trans_hybrid_term, info] = o3d.pipelines.odometry.compute_rgbd_odometry(source_image_depth,
                                                                                              target_image_depth,
                                                                                              intrinsic_matrix_target,
                                                                                              odometry_init,
                                                                                              o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                                                                                              odometry_option)
# source2 -> target
[success_hybrid_term_2, trans_hybrid_term_2, info_2] = o3d.pipelines.odometry.compute_rgbd_odometry(source_2_image_depth,
                                                                                                    target_image_depth,
                                                                                                    intrinsic_matrix_target,
                                                                                                    odometry_init_2,
                                                                                                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                                                                                                    odometry_option)
# visualizations
if success_hybrid_term:
    print("Local Registration: Source to Target Transformation: ", trans_hybrid_term)

    source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(source_image_depth, intrinsic_matrix_source)

    if not COLAB_FLAG:
        o3d.visualization.draw_geometries([target_point_cloud, source_pcd_hybrid_term],
                                          zoom=0.65,
                                          front=[0, 0, -1],
                                          lookat=[0, 0, 1.5],
                                          up=[0, -1, 0])

    # now do the transformation
    source_pcd_hybrid_term.transform(trans_hybrid_term)

    if not COLAB_FLAG:
        o3d.visualization.draw_geometries([target_point_cloud, source_pcd_hybrid_term],
                                          zoom=0.65,
                                          front=[0, 0, -1],
                                          lookat=[0, 0, 1.5],
                                          up=[0, -1, 0])

if success_hybrid_term_2:
    print("Local Registration: Source_2 to Target Transformation: ", trans_hybrid_term_2)

    source_pcd_hybrid_term_2 = o3d.geometry.PointCloud.create_from_rgbd_image(source_2_image_depth, intrinsic_matrix_source_2)

    if not COLAB_FLAG:
        o3d.visualization.draw_geometries([target_point_cloud, source_pcd_hybrid_term_2],
                                          zoom=0.65,
                                          front=[0, 0, -1],
                                          lookat=[0, 0, 1.5],
                                          up=[0, -1, 0])

    # now do the transformation
    source_pcd_hybrid_term_2.transform(trans_hybrid_term_2)

    if not COLAB_FLAG:
        o3d.visualization.draw_geometries([target_point_cloud, source_pcd_hybrid_term_2],
                                          zoom=0.65,
                                          front=[0, 0, -1],
                                          lookat=[0, 0, 1.5],
                                          up=[0, -1, 0])
