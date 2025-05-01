import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure, sphere_ball_intersection
import pyvista as pv
import argparse
import os
import pymeshfix

parser = argparse.ArgumentParser()
parser.add_argument('--colab_environment', type=str, required=False, help='present if in Colab Environment')
parser.add_argument('--save_output_path', type=str, required=False, help='path to save output images to')

parser.add_argument('--image_path', type=str, required=True, help='path to image')
parser.add_argument('--depth_path', type=str, required=True, help='path to depth map')
parser.add_argument('--intrinsics_path', type=str, required=True, help='path to intrinsics')
parser.add_argument('--prediction_map', type=str, required=True, help='path to segmentation prediction map')

args = parser.parse_args()

# if in Google Colab environment, set flag
COLAB_FLAG = False
SAVE_OUTPUT_FLAG = False

if args.colab_environment is not None:
    COLAB_FLAG = True

if args.save_output_path is not None:
    SAVE_OUTPUT_FLAG = True


# helper method to save open3D interactive outputs as images
# sources from https://stackoverflow.com/questions/62122925/pointcloud-to-image-in-open3d and
# https://github.com/isl-org/Open3D/issues/1110
def save_output(path, obj, zoom, lookat, up, front):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    vis.add_geometry(obj)

    ctrl = vis.get_view_control()

    ctrl.set_lookat(lookat)
    ctrl.set_zoom(zoom)
    ctrl.set_up(up)
    ctrl.set_front(front)

    vis.update_geometry(obj)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(path)
    vis.destroy_window()

    del ctrl
    del vis


# image needs to be RGB, so 3 channels; depth is 1 channel
# however, the output from the network is 4 channels (RGBA), in range [0, 255]
image = Image.open(args.image_path).convert('RGB')
depth = o3d.io.read_image(args.depth_path)

# get array representations
image_arr = np.asarray(image).astype(np.float32) / 255
depth_arr = np.asarray(depth).astype(np.float32)

# convert to Open3D image object
image = o3d.geometry.Image(image_arr)

# prediction map is also RGBA from the network, in range [0, 255]
pred_map = Image.open(args.prediction_map).convert('RGB')

intrinsics = np.load(args.intrinsics_path)
intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsics[0][0],
                                                         intrinsics[1][1],
                                                         intrinsics[0][2],
                                                         intrinsics[1][2])

image_depth_combined = o3d.geometry.RGBDImage.create_from_color_and_depth(image, depth)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image_depth_combined, intrinsics)

pred_map = np.asarray(pred_map).astype(np.float32) / 255

# figure out how many labels there are by counting unique colors
# source: https://stackoverflow.com/questions/24780697/numpy-unique-list-of-colors-in-the-image
# flatten all but last dimension
labels = np.unique(pred_map.reshape(-1, pred_map.shape[-1]), axis=0)

# list of partial pcds to be visualized
label_pcd_list = []

# go through each label to create a point cloud of points from this label
for label in labels:

    segmentation_map = np.where((pred_map == label).all(axis=2), 1, 0)
    depth_segmented_arr = (segmentation_map * depth_arr).astype(np.float32)

    segmentation_map = np.stack((segmentation_map, segmentation_map, segmentation_map), axis=-1)
    image_segmented_arr = (segmentation_map * image_arr).astype(np.float32)

    # convert back to Open3D objects
    image_segmented = o3d.geometry.Image(image_segmented_arr)
    depth_segmented = o3d.geometry.Image(depth_segmented_arr)

    segmented_image_depth_combined = o3d.geometry.RGBDImage.create_from_color_and_depth(image_segmented,
                                                                                        depth_segmented)
    segmented_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(segmented_image_depth_combined, intrinsics)

    # painting uniform colors: http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
    # don't paint background
    if not (label == 0).all():
        segmented_pcd = segmented_pcd.paint_uniform_color(label)
    label_pcd_list.append(segmented_pcd)

if not COLAB_FLAG:
    o3d.visualization.draw_geometries(label_pcd_list,
                                      zoom=0.65,
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 1.5],
                                      up=[0, -1, 0])

# if in Colab environment, cannot save images
if not COLAB_FLAG and SAVE_OUTPUT_FLAG:
    total_pcd = o3d.geometry.PointCloud()
    for item in label_pcd_list:
        total_pcd = total_pcd + item

    save_file_name = os.path.join(args.save_output_path, 'pcd_model.png')
    save_output(save_file_name, total_pcd, zoom=0.65, lookat=[0, 0, 1.5], front=[0, 0, -1], up=[0, -1, 0])

# perform surface fitting via ball-pivot algorithm
radii = [0.001, 0.005, 0.01, 0.02, 0.04, 0.08]

# iterate through each labeled pcd to perform surface fitting
USE_POISSON_FLAG = False

reconstruction_list = []
for i in range(1, len(label_pcd_list)):
    cur_pcd = label_pcd_list[i]
    cur_pcd = cur_pcd.voxel_down_sample(0.005)
    cur_pcd.estimate_normals()
    # http://www.open3d.org/docs/0.7.0/tutorial/Basic/pointcloud.html - normal orientation
    cur_pcd.orient_normals_towards_camera_location()
    if not USE_POISSON_FLAG:
        reconstruction = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cur_pcd, o3d.utility.DoubleVector(radii))

        # remove noisy triangles: http://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Connected-components
        triangle_clusters, cluster_n_triangles, _ = reconstruction.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 5
        reconstruction.remove_triangles_by_mask(triangles_to_remove)
    else:
        # http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
        reconstruction, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cur_pcd, depth=10)
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        reconstruction.remove_vertices_by_mask(vertices_to_remove)

    reconstruction = reconstruction.paint_uniform_color(labels[i])
    reconstruction_list.append(reconstruction)

if not COLAB_FLAG:
    o3d.visualization.draw_geometries(reconstruction_list,
                                      zoom=0.1,
                                      front=[-3, -2, -16],
                                      lookat=[0, 0, 0],
                                      up=[0, -1, 0],
                                      mesh_show_back_face=True)

total_reconstruction = o3d.geometry.TriangleMesh()
for item in reconstruction_list:
    total_reconstruction = total_reconstruction + item

if not COLAB_FLAG and SAVE_OUTPUT_FLAG:
    save_file_name = os.path.join(args.save_output_path, 'surface_reconstruction_model.png')
    save_output(save_file_name, total_reconstruction, zoom=0.1, lookat=[0, 0, 0], front=[-3, -2, -16], up=[0, -1, 0])

# create pyvista plotter object
if not COLAB_FLAG:
    plotter = pv.Plotter()

# calculate posture indices for each body region
for index in range(len(reconstruction_list)):
    # use Trimesh mesh
    # sources: https://stackoverflow.com/questions/56965268/how-do-i-convert-a-3d-point-cloud-ply-into-a-mesh-with-faces-and-vertices
    # and https://github.com/mikedh/trimesh/blob/main/examples/curvature.ipynb

    mesh = reconstruction_list[index]
    mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_normals))

    # clean up holes in the mesh using pymeshfix: https://pymeshfix.pyvista.org/
    mesh_fix = pymeshfix.PyTMesh()
    mesh_fix.load_array(mesh.vertices, mesh.faces)
    mesh_fix.fill_small_boundaries()
    vertices, faces = mesh_fix.return_arrays()
    mesh = trimesh.Trimesh(vertices, faces)

    # it seems we want smaller radii for better results
    r = 0.05
    mean = np.array(discrete_mean_curvature_measure(mesh, mesh.vertices, r)/sphere_ball_intersection(1, r))

    # mean curvatures can be negative or positive, so take absolute value
    mean = np.absolute(mean)

    # average the curvatures
    print("Average Mean Curvature For Surface " + str(index) + ": " + str(np.mean(mean)))

    if not COLAB_FLAG:
        plotter.add_mesh(pv.wrap(mesh), scalars=mean)

if not COLAB_FLAG:
    plotter.camera_position = (0, -0.1, -1)

# https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.save_graphic.html
if not COLAB_FLAG and SAVE_OUTPUT_FLAG:
    save_file_name = os.path.join(args.save_output_path, 'reconstruction_mean_curvature.svg')
    plotter.save_graphic(save_file_name)

if not COLAB_FLAG:
    plotter.show()
