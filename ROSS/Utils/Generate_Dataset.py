from pioneer.das.api.platform import Platform
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pioneer.common import platform, linalg
import math
from imantics import Polygons, Mask
from mmseg.apis import inference_model, init_model, show_result_pyplot
import matplotlib.path as mpltPath
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
import random
from collections import Counter
import alphashape
from scipy.spatial import ConvexHull

import numpy as np
from scipy.spatial import Delaunay

def create_large_sphere(radius=0.5):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)  # Adjust radius for visibility
    sphere.paint_uniform_color([1, 0, 0])  # Red color
    sphere.translate([0, 0, 0])  # Place at the origin
    return sphere

def ray_plane_intersection(origin, direction, plane_points):
    """
    Find the intersection of a ray and a plane defined by 3 points.

    :param origin: The origin of the ray (0,0,0).
    :param direction: The direction vector of the ray.
    :param plane_points: A 3x3 numpy array where each row represents a point on the plane.

    :return: The intersection point in 3D space and the reflection vector; otherwise, None.
    """
    # Plane definition
    p1, p2, p3 = plane_points

    # Normal to the plane
    normal = np.cross(p2 - p1, p3 - p1)
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector

    # Ensure the direction is not parallel to the plane
    denominator = np.dot(normal, direction)
    if np.abs(denominator) < 1e-6:
        return None, None  # No intersection, the ray is parallel to the plane

    # Ray-Plane intersection formula
    d = np.dot(normal, p1)
    t = d / denominator
    intersection = origin + t * direction

    # Calculate the reflection vector
    reflection_direction = direction - 2 * np.dot(direction, normal) * normal

    return intersection, reflection_direction

def point_in_triangle(pt, tri_pts):
    """Check if a point is inside the triangle formed by three points."""
    # Barycentric coordinates method
    x, y = pt
    x1, y1, x2, y2, x3, y3 = tri_pts.flatten()

    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    c = 1 - a - b

    return (0 <= a <= 1) and (0 <= b <= 1) and (0 <= c <= 1)


def find_closest_triangle(point_cloud, target_point):

    # Calculate distances from the target point to all points in the cloud
    distances = np.linalg.norm(point_cloud - target_point, axis=1)

    # Sort points by distance to the target point
    sorted_indices = np.argsort(distances)

    # Iterate over combinations of the closest points to find the first valid triangle
    for i in range(len(point_cloud) - 2):
        for j in range(i + 1, len(point_cloud) - 1):
            for k in range(j + 1, len(point_cloud)):
                triangle_indices = [sorted_indices[i], sorted_indices[j], sorted_indices[k]]
                triangle = point_cloud[triangle_indices]
                # Calculate the triangle's area
                area = 0.5 * np.abs(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
                if point_in_triangle(target_point, triangle) and area > 0.5:
                    #print(area)
                    return triangle_indices

    return None  # If no valid triangle is found

def create_ray(elevation, azimuth):

    # Convert to radians
    elevation = np.radians(elevation)
    azimuth = np.radians(azimuth)

    # Compute Cartesian coordinates
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)

    y=-y

    return np.array([x,y,z])

def Projection_Mask(point_Cloud):
    # Extract x, y, z coordinates
    x = point_Cloud[:, 0]
    y = -point_Cloud[:, 1]
    z = point_Cloud[:, 2]

    # Calculate azimuth (horizontal angle) and elevation (vertical angle)
    azimuth = np.arctan2(y, x)  # Azimuth angle in radians
    elevation = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))  # Elevation angle in radians

    # Convert radians to degrees
    azimuth = np.degrees(azimuth)
    elevation = np.degrees(elevation)

    keep = np.where((azimuth > -FOV * 0.5) & (azimuth < FOV * 0.5))[0]
    elevation = elevation[keep]
    keep2 = np.where((elevation > -FOV_Horizontal * 0.5) & (elevation < FOV_Horizontal * 0.5))[0]

    # Get the mask of the 2D points:
    azimuth = azimuth[keep][keep2]
    elevation = elevation[keep2]
    point_Cloud = point_Cloud[keep][keep2]

    # Get the mask of the 2D points:
    alpha = 0.1
    edge_points = alphashape.alphashape(np.array([azimuth, elevation]).T, alpha)

    try:
        edge_points = edge_points.exterior.coords.xy
    except:
        hull = ConvexHull(np.array([azimuth, elevation]).T)
        edge_points = np.array([azimuth[hull.vertices], elevation[hull.vertices]])


    return edge_points, point_Cloud , azimuth, elevation


# Load MMDetection Road Segmentation:
config_file = 'segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'
checkpoint_file = 'segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')


def distance_2D(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# ---------------------------------------
# ------------------------------------------------------------------------------
# TO DO !!!!!
# ----------------------------------------------------------------------------------------------------------------------

# Dataset path:
Dataset = '/home/watercooledmt/Datasets/Pixset/Raw_Data_exctracted/20200618_175654_part15_1380_1905'

pf = Platform(Dataset, progress_bar=False)

# ----------------------------------------------------------------------------------------------------------------------
#                                                      LABELS
# ----------------------------------------------------------------------------------------------------------------------
CLASSES = ['pedestrian', 'deformed pedestrian', 'bicycle', 'car', 'van', 'bus', 'truck',
           'motorcycle', 'stop sign', 'traffic light', 'traffic sign', 'traffic cone', 'fire hydrant',
           'guard rail', 'pole', 'pole group', 'road', 'sidewalk', 'wall', 'building', 'vegetation',
           'terrain',
           'ground', 'crosstalk', 'noise', 'others', 'animal', 'unpainted', 'cyclist', 'motorcyclist',
           'unclassified vehicle', 'obstacle', 'trailer', 'barrier', 'bicycle rack', 'construction vehicle','Unknown']

COLOR = [(176, 242, 182), (9, 82, 40), (255, 127, 0), (119, 181, 254), (15, 5, 107), (206, 206, 206),
         (91, 60, 17), (88, 41, 0), (217, 33, 33), (255, 215, 0), (48, 25, 212), (230, 110, 60),
         (240, 0, 32), (140, 120, 130), (80, 120, 130), (80, 120, 180), (255, 0, 0), (30, 70, 30),
         (230, 230, 130), (230, 130, 130), (60, 250, 60), (100, 140, 40), (100, 40, 40), (250, 10, 10),
         (250, 250, 250), (128, 128, 128), (250, 250, 10), (255, 255, 255), (198, 238, 242),
         (100, 152, 255),
         (50, 130, 200), (100, 200, 50), (255, 150, 120), (100, 190, 240), (20, 90, 200), (80, 40, 0),
         (128, 128, 128), (0, 0, 0)]

# ----------------------------------------------------------------------------------------------------------------------
#                                                  Parameters
# ----------------------------------------------------------------------------------------------------------------------
# Radar parameters
Range = 50
FOV = 120
FOV_Horizontal = 28

# Cylinder projection parameters
image_width = 2000
image_height = 500
FOV_width = 3.6651914291880923
FOV_height = 1.1780972450961724

# Road Detection:
Road_detection_treshold = 0.35

# DBscan
eps = 0.5
min_samples = 5
Nb_min_of_points_for_objects = 20
Do_Min_hight_for_DBscan = False
Min_hight = 0.25

show_DBSCAN = False

Vertical_Ray_numb = 20
Horizontal_Ray_numb = 20

# ----------------------------------------------------------------------------------------------------------------------
#                                                      Sensors
# ----------------------------------------------------------------------------------------------------------------------

# to synchronize Frame with Ouster points and bounding boxes:
spf = pf.synchronized(['pixell_bfc_box3d-deepen', 'flir_bfc_img-cyl','flir_bfc_poly2d-detectron-cyl'], ['ouster64_bfc_xyzit'], 2e3)


number_of_frame = len(spf)

fig = plt.figure(figsize=(20, 12))
for frame in range(number_of_frame):

    if frame ==0:
        continue
    Images = spf[frame]['flir_bfc_img-cyl']

    # Save image:
    #plt.imsave('image.png', Images.raw)

    # ------------------------------------------------------------------------------------------
    #                     STEP 0: POINT CLOUD FROM LIDAR
    # ------------------------------------------------------------------------------------------
    # ===== Lidar points =====
    Points_Lidar_GT = spf[frame]['ouster64_bfc_xyzit'].point_cloud()
    Distance = (Points_Lidar_GT[:, 0] ** 2 + Points_Lidar_GT[:, 1] ** 2) ** 0.5
    # Range
    Points_Lidar_GT = Points_Lidar_GT[Distance < Range, :]
    # Horizontal FOV
    angles_radians = np.arctan2(Points_Lidar_GT[:, 1], Points_Lidar_GT[:, 0])
    angles_degrees = np.degrees(angles_radians)
    Points_Lidar_GT = Points_Lidar_GT[(angles_degrees > -FOV / 2) & (angles_degrees < FOV / 2), :]

    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(Points_Lidar_GT)
    #o3d.visualization.draw_geometries([pcd])

    # ------------------------------------------------------------------------------------------
    #                     STEP 0: SAVE POINTS FROM OBJECT BOUNDING BOXES
    # ------------------------------------------------------------------------------------------
    boxes_sample = spf[frame]['pixell_bfc_box3d-deepen']

    # Manual projection of 3D points on the image:
    BOXES_3D = []
    Point_Cloud_3D = []
    Point_cloud_3D_Amplitudes = []
    Classified_points = []
    Classified_Class = []
    color_obj = []
    Class=[]

    for i in range(len(boxes_sample.raw['data'])):

        x, y, z = boxes_sample.raw['data']['c'][i]  # x, y, z coordinate of the center of the boxe
        # print('Distance from the sensor:',Distance_from_Sensor)
        l, L, H = boxes_sample.raw['data']['d'][i]  # length, width ,height
        beta, gamma, alpha = boxes_sample.raw['data']['r'][i]  # rotation around x,y,z
        id = boxes_sample.raw['data']['id'][i]  # the object instance unique ID
        classe = boxes_sample.raw['data']['classes'][i]  # The object category number
        Flag = boxes_sample.raw['data']['flags'][i]  # Miscellaneous infos

        # Define the rotation matrices around each axis
        alpha_rotation_matrix = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                                          [np.sin(alpha), np.cos(alpha), 0],
                                          [0, 0, 1]])
        beta_rotation_matrix = np.array([[1, 0, 0],
                                         [0, np.cos(beta), -np.sin(beta)],
                                         [0, np.sin(beta), np.cos(beta)]])
        gamma_rotation_matrix = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                                          [np.sin(gamma), np.cos(gamma), 0],
                                          [0, 0, 1]])

        # Define the rotation matrix
        rotation_matrix = np.dot(alpha_rotation_matrix, np.dot(beta_rotation_matrix, gamma_rotation_matrix))

        # Define the dimensions of the box
        dimensions = np.array([l, L, H])

        # Define the 8 corner points of the box
        corner_points = np.array([[dimensions[0] / 2, -dimensions[1] / 2, -dimensions[2] / 2],
                                  [dimensions[0] / 2, dimensions[1] / 2, -dimensions[2] / 2],
                                  [-dimensions[0] / 2, dimensions[1] / 2, -dimensions[2] / 2],
                                  [-dimensions[0] / 2, -dimensions[1] / 2, -dimensions[2] / 2],
                                  [dimensions[0] / 2, -dimensions[1] / 2, dimensions[2] / 2],
                                  [dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2],
                                  [-dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2],
                                  [-dimensions[0] / 2, -dimensions[1] / 2, dimensions[2] / 2]])

        # Rotate and translate the corner points to their final position
        corner_points = np.dot(rotation_matrix, corner_points.T).T
        corner_points[:, 0] += x
        corner_points[:, 1] += y
        corner_points[:, 2] += z

        # Save 3D point cloud for each box

        # change of base x,y to x',y',alpha:
        ptsXpYp = np.transpose(
            np.array([Points_Lidar_GT[:, 0] * np.cos(alpha) + Points_Lidar_GT[:, 1] * np.sin(alpha),
                      Points_Lidar_GT[:, 1] * np.cos(alpha) - Points_Lidar_GT[:, 0] * np.sin(alpha)]))
        xp, yp = x * np.cos(alpha) + y * np.sin(alpha), y * np.cos(alpha) - x * np.sin(alpha)

        # Keep Lidar points in the boxe
        keep = np.where(
            (ptsXpYp[:, 0] >= xp - l / 2) & (ptsXpYp[:, 0] <= xp + l / 2) & (ptsXpYp[:, 1] >= yp - L / 2) & (
                    ptsXpYp[:, 1] <= yp + L / 2))
        if len(keep[0]) >= 1:  # Lidar Points Find XY
            Z = np.transpose(Points_Lidar_GT[keep, 2])
            keep2 = np.where((Z >= z - H / 2) & (Z <= z + H / 2))
            if len(keep2[0]) > 10:
                keep = keep[0][keep2[0]]  # Lidar points in the boxe
                # Save 3D point cloud for each box
                Point_Cloud_3D.append(Points_Lidar_GT[keep, :])
                # Save 3D point cloud amplitudes for each box
                Point_cloud_3D_Amplitudes.append(spf[frame]['ouster64_bfc_xyzit'].amplitudes[keep])

                # Save the box:
                BOXES_3D.append(corner_points)
                color_obj.append(COLOR[classe])

                # Save the classified points indexes:
                Classified_points.append(keep)
                Classified_Class.append(classe)

                if classe not in Class:
                    Class.append(classe)

    # Get the unclassified points
    Classified_points_index = np.concatenate(Classified_points)
    Unclassified_points_index = np.delete(np.arange(len(Points_Lidar_GT)), Classified_points_index)

    # -------------------------------------------------------------------------------------------------------------------------------------------
    #                                                                ROAD SEGMENTATION
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # MMSEGMENTATION:
    result = inference_model(model, Images.raw)
    #show_result_pyplot(model, Images.raw, result, show=True)
    result=result.pred_sem_seg.data.cpu().numpy()
    result = result[0]
    Road = np.where((result == 0) | (result == 1), True, False)

    MASQUE = [Road]
    MASQUE_LABEL = ['road']
    MASQUE_COLOR = [[155.0, 155.0, 155.0]]

    Nb_mask = 0
    MASQUE_POINTS = []
    Label = []
    Color = []
    res_borderlessTable = []
    for i in range(len(MASQUE)):
        for j in range(len(Mask(MASQUE[i]).polygons().points)):
            MASQUE_POINTS.append(Mask(MASQUE[i]).polygons().points[j])
            Label.append(MASQUE_LABEL[i])
            Color.append(MASQUE_COLOR[i])
            res_borderlessTable.append(
                [min(Mask(MASQUE[i]).polygons().points[j][:, 0]),
                 min(Mask(MASQUE[i]).polygons().points[j][:, 1]),
                 max(Mask(MASQUE[i]).polygons().points[j][:, 0]),
                 max(Mask(MASQUE[i]).polygons().points[j][:, 1])])

    T = len(MASQUE_POINTS[0])
    Real_MP = 0
    for i in range(len(MASQUE_POINTS)):
        if len(MASQUE_POINTS[i]) > T:
            T = len(MASQUE_POINTS[i])
            Real_MP = i

    coord = MASQUE_POINTS[Real_MP].tolist()
    xs, ys = zip(*coord)  # create lists of x and y values
    xs = np.array(xs)
    ys = np.array(ys)

    # Project the unclassified points in the camera referential
    Transform_matrix = spf[frame]['ouster64_bfc_xyzit'].compute_transform('flir_bfc')
    point_cloud_in_camera_ref = linalg.map_points(Transform_matrix, Points_Lidar_GT[Unclassified_points_index])

    # API projection method:
    image_sample = spf[frame]['flir_bfc_img-cyl']
    # plot_image(image_sample)
    pts2d = image_sample.project_pts(point_cloud_in_camera_ref)

    # -------------------------------------------------------------------------------------------
    # STEP 1: GET RID OF POINTS OUTSIDE THE IMAGE
    # -------------------------------------------------------------------------------------------

    # We keep only the points inside the image:
    keep = np.where((pts2d[:, 0] > 0) & (pts2d[:, 0] < image_sample.raw.shape[1] - 1) & (pts2d[:, 1] > 0) & (
            pts2d[:, 1] < image_sample.raw.shape[0] - 1))[0]
    Points_to_keep = keep
    # -------------------------------------------------------------------------------------------
    # STEP 2: GET RID OF POINTS OUTSIDE THE ROAD MASK (to run faster)
    # -------------------------------------------------------------------------------------------

    # Find higest point in the RoadMask and keep points above:
    Min_high = min(np.array(coord)[:, 1])
    keep = np.where(pts2d[Points_to_keep, 1] < Min_high)[0]
    keep2 = np.where(pts2d[Points_to_keep, 1] >= Min_high)[0]

    Points_to_Test = Points_to_keep[keep2]
    Points_to_keep = Points_to_keep[keep]

    Points_to_Test_array = np.array(Points_to_Test)
    KEEP_Tested_Points = Points_to_Test_array[
        ~mpltPath.Path(coord).contains_points(pts2d[Points_to_Test_array])]
    KEEP_Road_point = Points_to_Test_array[mpltPath.Path(coord).contains_points(pts2d[Points_to_Test_array])]

    Points_to_keep = list(Points_to_keep) + list(KEEP_Tested_Points)

    Points_to_test = Points_Lidar_GT[Unclassified_points_index][Points_to_keep]
    Road_Points_GT = Points_Lidar_GT[Unclassified_points_index][KEEP_Road_point]

    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(Road_Points_GT)
    #o3d.visualization.draw_geometries([pcd])

    # -------------------------------------------------------------------------------------------
    # STEP 4: CLEAR UNSAVED POINTS:
    # -------------------------------------------------------------------------------------------

    # Unsaved Index:
    All_index = np.linspace(0, len(Points_to_test[:, 0]) - 1, len(Points_to_test[:, 0]))
    Unsaved_Index = All_index.astype(int)


    # USING ROAD PLANE EQUATION TO FIND ALL REMAINING POINTS FROM THE ROAD.
    # ---------------------------------------------------------------------


    # Generate some sample data
    x = Road_Points_GT[:, 0]
    y = Road_Points_GT[:, 1]
    z = Road_Points_GT[:, 2]
    X = np.column_stack((x, y, np.ones_like(x)))

    model_2 = LinearRegression().fit(X, z)

    Linear_fit_a = model_2.coef_[0]  # coefficient of x
    Linear_fit_b = model_2.coef_[1]  # coefficient of y
    Linear_fit_c = -1  # coefficient of z
    Linear_fit_d = model_2.intercept_  # constant term

    Distance= Points_to_test[:, 2] - (Linear_fit_a * Points_to_test[:, 0] + Linear_fit_b * Points_to_test[:, 1] + Linear_fit_d)

    keep = np.where(Distance < Road_detection_treshold)[0]

    # Add the points to the saved points:
    Road_point_index=list(Unclassified_points_index[KEEP_Road_point]) + list(Unclassified_points_index[Points_to_keep][keep])
    Road_point_index=np.array(Road_point_index).astype(int)

    Point_to_test_index = np.delete(Unclassified_points_index[Points_to_keep], keep).astype(int)

    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(Points_Lidar_GT[Point_to_test_index])
    #o3d.visualization.draw_geometries([pcd])


    # ---------------------------------------------------------------------
    # DBSCAN Clustering:
    # ---------------------------------------------------------------------
    XYZ = Points_Lidar_GT[Point_to_test_index]

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(XYZ)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    cmap = plt.cm.get_cmap('hsv', lut=n_clusters_)
    n_noise_ = list(labels).count(-1)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    # open 3D DBscan

    points = [0, 0, 0]
    color = [0, 0, 0]
    State = []
    nb_of_cluster = 0
    Index = [[0]]
    L = np.linspace(0, len(colors) - 1, len(colors)).astype(int)
    random.shuffle(L)
    KEEP_DBSCAN = []
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        else:
            if (Counter(labels)[k] > Nb_min_of_points_for_objects):
                class_member_mask = labels == k
                xyz = XYZ[class_member_mask & core_samples_mask]
                # c = np.array([tuple(col)])[0][0:3]
                #if Do_Min_hight_for_DBscan:
                    #Distance_moyenne = xyz[:, 2] - np.min(xyz[:, 2])
                    #keep = len(np.where(Distance_moyenne > Min_hight)[0]) / len(Distance_moyenne)
                    #if (keep > 0.7):
                keep = np.where(class_member_mask & core_samples_mask == True)[0]
                if len(keep) > Nb_min_of_points_for_objects:
                    KEEP_DBSCAN.append(keep)
                    points1 = np.vstack((xyz[:, 0], xyz[:, 1], xyz[:, 2])).transpose()
                    color1 = np.zeros((len(points1), 3)) + list(cmap(L[k])[0:3])
                    points = np.c_[points, points1.transpose()]
                    color = np.c_[color, color1.transpose()]
                    Index.append(list(np.zeros(len(points1)) + nb_of_cluster))
                    nb_of_cluster += 1

    # Assuming points is a Nx3 numpy array with columns representing x, y, z coordinates
    #points = Points_Lidar_GT
    #color_3D = np.zeros((len(points), 3))  # default to black
    #color_3D[Road_point_index] = [225, 225, 225]  # Road points in gray
    #for i in range(len(Classified_points)):
        #color_3D[Classified_points[i]] = color_obj[i]

    Points_Lidar_GT[:, 2] = Points_Lidar_GT[:, 2] + 0.547

    Data_Polygons = []
    Data_point_Cloud = []
    Data_azimuth = []
    Data_elevation = []
    Data_Type=[]

    # Road points:
    Polygons, point_Cloud, azimuth, elevation = Projection_Mask(Points_Lidar_GT[Road_point_index])
    Data_Polygons.append(Polygons)
    Data_point_Cloud.append(point_Cloud)
    Data_azimuth.append(azimuth)
    Data_elevation.append(elevation)
    Data_Type.append('Road')

    # Unclassified points:
    for i in range(len(KEEP_DBSCAN)):
        #print(Points_Lidar_GT[Point_to_test_index[KEEP_DBSCAN[i]]])
        Polygons, point_Cloud, azimuth, elevation = Projection_Mask(Points_Lidar_GT[Point_to_test_index[KEEP_DBSCAN[i]]])
        Data_Polygons.append(Polygons)
        Data_point_Cloud.append(point_Cloud)
        Data_azimuth.append(azimuth)
        Data_elevation.append(elevation)
        Data_Type.append('Unknown object')

    for i in range(len(Classified_points)):
        Polygons, point_Cloud, azimuth, elevation = Projection_Mask(Points_Lidar_GT[Classified_points[i]])
        Data_Polygons.append(Polygons)
        Data_point_Cloud.append(point_Cloud)
        Data_azimuth.append(azimuth)
        Data_elevation.append(elevation)
        Data_Type.append(CLASSES[Classified_Class[i]])

    Vertical_Range=np.linspace(-FOV_Horizontal/2,FOV_Horizontal/2,Vertical_Ray_numb)
    Horizontal_Range=np.linspace(-FOV/2,FOV/2,Horizontal_Ray_numb)

    Final_Saved_3D_points = []
    Final_Saved_Ray_points = []
    Final_Saved_Distance = []
    Final_Saved_Type = []
    Final_Saved_Angle_of_reflection = []

    for i in range(len(Vertical_Range)-1):
        for j in range(len(Horizontal_Range)-1):
            X_Beam=Horizontal_Range[j]+(Horizontal_Range[1]-Horizontal_Range[0])/2
            Y_Beam=Vertical_Range[i]+(Vertical_Range[1]-Vertical_Range[0])/2


    for i in range(len(Vertical_Range)-1):
        for j in range(len(Horizontal_Range)-1):
            X_Beam=Horizontal_Range[j]+(Horizontal_Range[1]-Horizontal_Range[0])/2
            Y_Beam=Vertical_Range[i]+(Vertical_Range[1]-Vertical_Range[0])/2
            Inside_Index=[]
            for k in range(len(Data_Polygons)):
                if mpltPath.Path(np.array(Data_Polygons[k]).T).contains_points([[X_Beam,Y_Beam]]):
                    Inside_Index.append(k)
            if len(Inside_Index)>0:
                Saved_3D_points=[]
                Saved_Ray_points=[]
                Saved_Distance=[]
                Saved_Type=[]
                Saved_Angle_of_reflection=[]
                for k in Inside_Index:
                    Index=find_closest_triangle(np.array([Data_azimuth[k],Data_elevation[k]]).T,[X_Beam,Y_Beam])
                    #(Data_Type[k])
                    #print(Index)
                    if Index is not None:
                        # Example usage
                        plane_points = Data_point_Cloud[k][Index]

                        elevation_angle = Y_Beam
                        horizontal_angle = X_Beam

                        origin = np.array([0, 0, 0])
                        direction = create_ray(elevation_angle, horizontal_angle)
                        #print(plane_points)
                        intersection_point, angle_of_reflection = ray_plane_intersection(origin, direction, plane_points)
                        if intersection_point is not None:
                            Distance = np.linalg.norm(intersection_point)
                            Saved_Ray_points.append(intersection_point)
                            Saved_3D_points.append(Data_point_Cloud[k][Index])
                            Saved_Distance.append(Distance)
                            Saved_Type.append(Data_Type[k])
                            Saved_Angle_of_reflection.append(angle_of_reflection/5)

                # Get index of min distance in Saved_Distance
                Closer_surface_index=np.argmin(Saved_Distance)

                # Save the closest point:
                Final_Saved_Ray_points.append(Saved_Ray_points[Closer_surface_index])
                Final_Saved_3D_points.append(Saved_3D_points[Closer_surface_index])
                Final_Saved_Distance.append(Saved_Distance[Closer_surface_index])
                Final_Saved_Type.append(Saved_Type[Closer_surface_index])
                Final_Saved_Angle_of_reflection.append(Saved_Angle_of_reflection[Closer_surface_index])


    Final_Saved_Ray_points=np.array(Final_Saved_Ray_points)
    Final_Saved_Ray_points_color = np.zeros((len(Final_Saved_Ray_points), 3))  # default to black
    Final_Saved_Ray_points_color[:, 0] = 1  # set the red channel to 1 for all points

    '''
    for i in range(len(Final_Saved_Type)):
        if Final_Saved_Type[i] == 'Road':
            Final_Saved_Ray_points_color[i] = [225, 225, 225]
        elif Final_Saved_Type[i] == 'Unknown object':
            Final_Saved_Ray_points_color[i] = [0, 0, 0]
        else:
            Class_index = CLASSES.index(Final_Saved_Type[i])
            Final_Saved_Ray_points_color[i] = COLOR[Class_index]
    '''

    pcd_Full = o3d.geometry.PointCloud()
    pcd_Full.points = o3d.utility.Vector3dVector(Points_Lidar_GT)


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Final_Saved_Ray_points)
    pcd.normals = o3d.utility.Vector3dVector(Final_Saved_Angle_of_reflection)
    pcd.colors = o3d.utility.Vector3dVector(Final_Saved_Ray_points_color)

    # Create a large sphere at the origin
    origin_sphere = create_large_sphere(radius=0.1)  # Adjust radius for visibility

    o3d.visualization.draw_geometries([pcd,pcd_Full,origin_sphere])




    #plt.figure()
    #plt.scatter(azimuth[Inside_FOV_Index], elevation[Inside_FOV_Index], s=4, c=color_3D[Inside_FOV_Index] / 255)
    #plt.pause(0.1)
    #plt.clf()

    # Road_Points



    '''

    #keep

    gs = gridspec.GridSpec(3, 1)

    # First subplot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(Images.raw)
    ax1.set_title('Image')
    ax1.axis('off')


    # Legend:
    Legend=[]
    for i in range(len(Class)):
        patch = mpatches.Patch(color=np.array(COLOR[Class[i]]).reshape(1, -1)/ 255, label=CLASSES[Class[i]])
        Legend.append(patch)

    # Add the road and unknown object to the legend:
    patch = mpatches.Patch(color='gray', label='Road')
    Legend.append(patch)
    patch = mpatches.Patch(color='k', label='Unknown object')
    Legend.append(patch)

    ax2 = fig.add_subplot(gs[1:, 0])
    #plot the road:
    ax2.scatter(-Points_Lidar_GT[Road_point_index, 1], Points_Lidar_GT[Road_point_index, 0], s=3, c='gray')
    #plot the points to test:
    ax2.scatter(-Points_Lidar_GT[Point_to_test_index, 1], Points_Lidar_GT[Point_to_test_index, 0], s=3, c='k')
    #plot the BB points:
    for i in range(len(Classified_points)):
        ax2.scatter(-Points_Lidar_GT[Classified_points[i], 1], Points_Lidar_GT[Classified_points[i], 0], s=4, c=np.array(color_obj[i]).reshape(1, -1) / 255)
    ax2.set_xlim([-Range * np.sin(np.pi * (180-FOV*0.5) / 180), Range * np.sin(np.pi * (180-FOV*0.5) / 180)])
    ax2.set_ylim([0, Range])
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Segmented and classified LIDAR points bird view')
    ax2.legend(handles=Legend)

    plt.tight_layout()
    plt.savefig('./Figures/frame'+str(frame)+'.png')
    plt.clf()


    if show_DBSCAN:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.transpose(points))
        pcd.colors = o3d.utility.Vector3dVector(np.transpose(color))
        o3d.visualization.draw_geometries([pcd])
    '''
