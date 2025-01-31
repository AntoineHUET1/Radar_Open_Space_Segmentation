import os
import random
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
from .Data_Loaders import Load_GT_Data,Load_Radar_Data,adjust_radar_data_based_on_range_and_gt
from ROSS.Utils.Generate_Graph import COLOR,CLASSES,Transform_matrix,image_width,image_height,FOV_width,FOV_height
import ROSS.cfg.ROSS_Config as cfg
import matplotlib.patches as mpatches
from ROSS.Utils.Test import Test
# Parameters:
Angular_rez=1
Angular_rez_scan=0.1
Limit_vehicle= -0.87 + 1.69 + 0.5 # Road limit + Vehicle height + Margin
Dataset_path='/home/watercooledmt/PycharmProjects/Radar_Open_Space_Segmentation/data/ROSS_Dataset/'
# Radar Spec:
Radar_FOV = 120
Radar_Range = 50


# Lines used as open space limit:
Save_Lines=False # Save the open space lines
Number_of_lines=5
angle_threshold=0.5

# Point Cloud Modification:
Remove_Out_of_Drivable_Space = True

CLASSES = ['pedestrian', 'deformed pedestrian', 'bicycle', 'car', 'van', 'bus', 'truck',
           'motorcycle', 'stop sign', 'traffic light', 'traffic sign', 'traffic cone', 'fire hydrant',
           'guard rail', 'pole', 'pole group', 'road', 'sidewalk', 'wall', 'building', 'vegetation',
           'terrain',
           'ground', 'crosstalk', 'noise', 'others', 'animal', 'unpainted', 'cyclist', 'motorcyclist',
           'unclassified vehicle', 'obstacle', 'trailer', 'barrier', 'bicycle rack', 'construction vehicle','Unknown']

COLOR = [(176, 242, 182), (9, 82, 40), (255, 127, 0), (119, 181, 254), (15, 5, 107), (206, 206, 206),
         (91, 60, 17), (88, 41, 0), (217, 33, 33), (255, 215, 0), (48, 25, 212), (230, 110, 60),
         (240, 0, 32), (140, 120, 130), (80, 120, 130), (80, 120, 180), (30, 30, 30), (30, 70, 30),
         (230, 230, 130), (230, 130, 130), (60, 250, 60), (100, 140, 40), (100, 40, 40), (250, 10, 10),
         (250, 250, 250), (128, 128, 128), (250, 250, 10), (255, 255, 255), (198, 238, 242),
         (100, 152, 255),
         (50, 130, 200), (100, 200, 50), (255, 150, 120), (100, 190, 240), (20, 90, 200), (80, 40, 0),
         (128, 128, 128), (255, 0, 0)]

def remove_small_gap(angle_step, range_step, angular_rez_max_gap=4):
    """
        Adjusts small angular gaps in open space limit by shifting angles and range values
        to smooth out consecutive readings that are too close to each other.

        Parameters:
        - angle_step: List of angle measurements.
        - range_step: List of range measurements corresponding to open space limit.
        - angular_rez_max_gap: Maximum allowed angular resolution (gap) between consecutive angles.
                               If the gap is smaller, adjustments will be made (default: 4).
        """
    n = len(angle_step)

    for i in range(n - 1):
        current_range = range_step[i]
        next_range = range_step[i + 1]

        # Check if both current and next range equal to Radar_Range
        if current_range == Radar_Range and next_range == Radar_Range:
            # Estimate angular resolution
            angular_rez = angle_step[i + 1] - angle_step[i]

            if angular_rez < angular_rez_max_gap:
                # Adjust the current point if it's not the first element
                if i > 0:
                    range_step[i] = range_step[i - 1]
                    angle_step[i] += angular_rez / 2  # Shift angle slightly

                # Adjust the next point if it's not the last element
                if i < n - 2:
                    range_step[i + 1] = range_step[i + 2]
                    angle_step[i + 1] -= angular_rez / 2  # Shift angle slightly

    return angle_step, range_step

def project_points(x, y,Road_Equation):

    z = Road_Equation[0]['a'] * x + Road_Equation[0]['b'] * y + Road_Equation[0]['c']

    point_cloud_in_camera_ref = map_points(Transform_matrix, np.transpose(np.array([x, y, z])))

    # project points inside the image:
    pts2 = point_cloud_in_camera_ref.T
    azimut = np.arctan2(pts2[0, :], pts2[2, :])
    norm_xz = np.linalg.norm(pts2[[0, 2], :], axis=0)
    elevation = np.arctan2(pts2[1, :], norm_xz)
    x = np.round(image_width / 2 + azimut * (image_width / FOV_width))
    y = np.round(image_height / 2 + elevation * (image_height / FOV_height))
    pts2d = np.column_stack((x, y))
    # Change last and first point to close the polygon
    pts2d[0, :] = [420, 500]
    pts2d[-1, :] = [1590, 500]

    # Resize the points to the original image size
    pts2d[:, 0] = pts2d[:, 0] - 420

    return pts2d

def remove_out_of_drivable_space(point_cloud, road_plan_equation,Limit_vehicle):
    """
    Removes points from the point cloud that are above a certain height limit from the drivable surface defined by the road plan equation.

    Parameters:
    - point_cloud: Numpy array of shape (N, 3), where N is the number of points. Each point is represented by (x, y, z) coordinates.
    - road_plan_equation: List of dictionaries containing the coefficients 'a', 'b', and 'c' that define the road plane equation (z = ax + by + c).
    - Limit_vehicle: Maximum height limit for the vehicle.

    Returns:
    - Filtered point_cloud: Numpy array of points that are within the drivable space.
    """

    # Extract the road plane coefficients (a, b, c) from the road_plan_equation dictionary
    a = road_plan_equation[0]['a']
    b = road_plan_equation[0]['b']
    c = road_plan_equation[0]['c']

    # Calculate the expected z-values (height) for the road plane at each (x, y) point in the point cloud
    z_road = a * point_cloud[:, 0] + b * point_cloud[:, 1] + c

    # Extract the actual z-values (height) of the points in the point cloud
    z = point_cloud[:, 2]

    # Calculate the height difference between each point and the road plane
    height = z - z_road

    # Identify the indices of points that are within the acceptable height limit (below Limit_vehicle)
    keep = np.where(height < Limit_vehicle)[0]

    # Return the filtered point cloud with only the points that are within the drivable space
    return point_cloud[keep]


def split_pc(pc, max_delta_angle=6):
    """
    Splits the point cloud into segments based on angular variation.
    If the angle difference between consecutive points exceeds the specified threshold,the point cloud is split into multiple parts.
    """

    # Convert point cloud to Range-Angle format and sort by angle
    range_values, angle, _ = Cathesian_to_RA(pc)
    idx = np.argsort(angle)
    angle, pc = angle[idx], pc[idx]

    # Find indices where the angle difference exceeds the threshold
    split_idx = [0]
    for i in range(len(angle) - 1):
        if angle[i + 1] - angle[i] > max_delta_angle:
            split_idx.append(i + 1)

    # Split the point cloud at identified indices
    if len(split_idx) > 1:
        split_idx.append(len(angle))
        return [pc[split_idx[i]:split_idx[i + 1]] for i in range(len(split_idx) - 1)]

    # Return the original point cloud if no split is necessary
    return [pc]

def map_points(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform on 3x1 point(s)

    Args:
        m: a 4x4 transform
        v: Nx3 point matrix
    """
    return (np.dot(m[:3, :3], v.T) + m[:3, 3, None]).T


def remove_outliers(lines, angle_threshold):
    """
    Removes outliers from the range values by interpolating and filtering ranges at specific angle steps.

    Parameters:
    - lines: List of 2D arrays where each array represents (angle, range) data.
    - angle_threshold: Step size for generating angles to interpolate and filter ranges.

    Returns:
    - interpolated_ranges: 2D array of (angle, average_filtered_range) after outlier removal.
    """

    # Step 1: Concatenate all angles from the lines to find the global min and max angles
    all_angles = np.concatenate([line[:, 0] for line in lines])
    min_angle, max_angle = np.min(all_angles), np.max(all_angles)

    # Step 2: Generate evenly spaced angles from min_angle to max_angle based on angle_threshold
    angle_steps = np.arange(min_angle, max_angle, angle_threshold)
    if max_angle not in angle_steps:  # Ensure max_angle is included
        angle_steps = np.append(angle_steps, max_angle)

    # Step 3: Interpolate range values for each angle in angle_steps
    interpolated_ranges = []
    for angle in angle_steps:
        ranges_at_angle = []

        # For each line, interpolate the range value at the current angle
        for line in lines:
            if min(line[:, 0]) <= angle <= max(line[:, 0]):  # Check if the angle is within the line's angle range
                interpolated_range = np.interp(angle, line[:, 0], line[:, 1])
                ranges_at_angle.append(interpolated_range)

        # Step 4: Filter out outliers using the standard deviation
        if ranges_at_angle:
            mean_range = np.mean(ranges_at_angle)
            std_dev_range = np.std(ranges_at_angle)
            threshold = 1  # Use 1 standard deviation for filtering
            filtered_ranges = [r for r in ranges_at_angle if abs(r - mean_range) <= threshold * std_dev_range]

            # Step 5: Save the angle and the average of the filtered ranges
            if filtered_ranges:
                avg_range = np.mean(filtered_ranges)
                interpolated_ranges.append([angle, avg_range])

    # Step 6: Return the final result as a 2D array of (angle, average_filtered_range)
    return np.array(interpolated_ranges)

# Function to plot the range-angle scatter plot
def plot_range_angle_scatter(point_cloud,Mode,Save_Lines,Road_Plan_Equation,frame,Sequence):
    Lines_to_save=[]
    # List of detected classes for legend
    detected_classes = []
    for i in range(len(point_cloud)):
        # Remove the road:
        if point_cloud[i]['class'] == 16 :
            continue

        # Process the point cloud and filter out non-drivable areas
        pc = point_cloud[i]['point_cloud']
        pc = remove_out_of_drivable_space(pc,Road_Plan_Equation,Limit_vehicle)
        if len(pc) == 0:
            continue

        # Split the unclassified class into subclasses if necessary:
        List_pc = split_pc(pc) if point_cloud[i]['class'] == 37 else [pc]

        for pc in List_pc:

            color = [0, 0, 0] if Mode == 0 else COLOR[point_cloud[i]['class']]
            Range, Angle, z = Cathesian_to_RA(pc)

            if not Save_Lines:
                plt.scatter(Angle,Range, c=np.array(color).reshape(1, -1) / 255, s=6)

            # Add the class to the detected classes list
            if point_cloud[i]['class'] not in detected_classes:
                detected_classes.append(point_cloud[i]['class'])

            # Vertical segmentation for OPEN SPACE detection
            if Mode ==0 :
                # Remove first  10% of the data
                Z_max, Z_min = np.max(z), np.min(z) + 0.1 * (np.max(z) - np.min(z))

                # Plot the point cloud
                plt.scatter(Angle, Range, c=np.array(color).reshape(1, -1) / 255, s=6)

                # Split the data into segments based on Z levels and
                Z_split = np.linspace(Z_min, Z_max, Number_of_lines)
                Lines=[]
                for j in range(len(Z_split)-1):
                    Keep_split =np.where((z>=Z_split[j]) & (z<Z_split[j+1]))[0]
                    if len(Keep_split)>1:
                        Range_split = Range[Keep_split]
                        Angle_split = Angle[Keep_split]

                        Line = [[Angle_split[np.argmin(Angle_split)], Range_split[np.argmin(Angle_split)]]]
                        for j in np.arange(min(Angle_split), max(Angle_split), Angular_rez):
                            keep = np.where((Angle_split >= j) & (Angle_split < j + Angular_rez))[0]
                            if len(keep) > 0:
                                min_index = np.argmin(Range_split[keep])
                                Line.append([Angle_split[keep[min_index]], Range_split[keep[min_index]]])

                        Line.append([Angle_split[np.argmax(Angle_split)], Range_split[np.argmax(Angle_split)]])
                        Line = np.array(Line)
                        Lines.append(Line)
                if len(Lines)>0:
                    # From the lines, remove outlier points and generate open space limit lines
                    Output_Line=remove_outliers(Lines,angle_threshold)
                    Lines_to_save.append({'class':point_cloud[i]['class'],'lines':Output_Line})


    if Mode ==0:
        # Sort and process detected lines for final ground truth (GT) generation
        Angle_step=[line_data['lines'] for line_data in Lines_to_save]
        # Flatten: `
        Angle_step = np.concatenate(Angle_step)
        # Sort:
        Angle_step = Angle_step[np.argsort(Angle_step[:, 0])]
        Angle_step = Angle_step[:, 0]
        Range_step,Angle_step_2,Line_idx=[],[],[]

        # Add boundary angles to the range
        if Radar_FOV/2 not in Angle_step:
            Angle_step = np.append(Angle_step, Radar_FOV/2)
        if -Radar_FOV/2 not in Angle_step:
            Angle_step = np.insert(Angle_step, 0, -Radar_FOV/2)

        # Interpolate the range values for each angle to keep the closest obstacle
        for i in range(len(Angle_step)):
            angle=Angle_step[i]
            ranges_at_angle = []
            line_idx=[]

            # Find the closest line value:
            for idx,line_data in enumerate(Lines_to_save):
                line=line_data['lines']
                if min(line[:, 0]) <= angle <= max(line[:, 0]):  # Ensure angle is within the line's range
                    interpolated_range = np.interp(angle, line[:, 0], line[:, 1])
                    ranges_at_angle.append(interpolated_range)
                    line_idx.append(idx)

            if ranges_at_angle:

                # Find the minimum index
                min_index_line = np.argmin(ranges_at_angle)
                Line_idx.append(line_idx[min_index_line])
                Range_step.append(ranges_at_angle[min_index_line])
                Angle_step_2.append(angle)

                # Check if the line as changed for better visualization of the open space
                if len(Line_idx)>1:
                    if Line_idx[-1]!=Line_idx[-2]:

                        ranges_at_angle_2 = []
                        for idx,line_data in enumerate(Lines_to_save):
                            if idx == Line_idx[-2]:
                                continue
                            line = line_data['lines']
                            if min(line[:, 0]) <= Angle_step[i-1] <= max(line[:, 0]):
                                ranges_at_angle_2.append(interpolated_range)
                        if len(ranges_at_angle_2) > 0:
                            min_index = np.argmin(ranges_at_angle_2)
                            Range_step.insert(-1, ranges_at_angle_2[min_index])

                            Angle_step_2.insert(-1, Angle_step[i-1])
                        else:
                            Range_step.insert(-1, Radar_Range)
                            Angle_step_2.insert(-1, Angle_step[i-1])
                            if i<len(Angle_step)-1:
                                Range_step.insert(-1, Radar_Range)
                                Angle_step_2.insert(-1, Angle_step[i])

                        ranges_at_angle_2 = []
                        for idx,line_data in enumerate(Lines_to_save):
                            if idx == Line_idx[-2]:
                                continue
                            line = line_data['lines']
                            if min(line[:, 0]) <= angle <= max(line[:, 0]):
                                ranges_at_angle_2.append(interpolated_range)
                        if len(ranges_at_angle_2) > 0:

                            min_index = np.argmin(ranges_at_angle_2)
                            Range_step.insert(-1, ranges_at_angle_2[min_index])
                            Angle_step_2.insert(-1, angle)

                        else:
                            Range_step.insert(-1, Radar_Range)
                            Angle_step_2.insert(-1, angle)


            else:
                Line_idx.append(999)
                if len(Range_step)>0:
                    Range_step.append(Range_step[-1])
                    Angle_step_2.append(angle)
                else:
                    Range_step.append(Radar_Range)
                    Angle_step_2.append(angle)

        # Add a value at the beginning and
        idx_Radar_range=0
        for i in range(len(Angle_step_2)):
            if Range_step[i] == Radar_Range:
                idx_Radar_range+=1
            else:
                break
        if abs(Angle_step_2[idx_Radar_range]-Angle_step_2[0])<4*Angular_rez:
            for i in range(idx_Radar_range):
                Range_step[i]=Range_step[idx_Radar_range+1]

        # Reverse
        idx_Radar_range = 0
        for i in range(len(Angle_step_2)):
            if Range_step[-(i+1)] == Radar_Range:
                idx_Radar_range += 1
            else:
                break
        if abs(Angle_step_2[-1] - Angle_step_2[-(idx_Radar_range+1)]) < 4 * Angular_rez:
            for i in range(idx_Radar_range):
                Range_step[-(i+1)] = Range_step[-(idx_Radar_range+2)]


        keep=np.where(np.array(Range_step)<Radar_Range)[0]
        #Stack Angle and Range in GT array:
        GT=np.vstack((np.array(Angle_step_2)[keep],np.array(Range_step)[keep])).T
        #save GT
        np.save(Dataset_path+Sequence+'/GT_2/'+str(frame)+'.npy',GT)

        Angle_step_2,Range_step=remove_small_gap(Angle_step_2,Range_step)

        # Add Unknown area:
        plt.gca().set_facecolor('lightgrey')

        # Fill the area inside the FOV in red

        # add the first and last point
        Angle_step_2.insert(0, -Radar_FOV/2)
        Range_step.insert(0, 0)
        Angle_step_2.append(Radar_FOV/2)
        Range_step.append(0)
        plt.fill(Angle_step_2, Range_step, c='lightcoral')

        # plot lidar points:
        for i in range(len(point_cloud)):
            if point_cloud[i]['class'] == 16:
                continue
            pc = point_cloud[i]['point_cloud']
            pc = remove_out_of_drivable_space(pc, Road_Plan_Equation,Limit_vehicle)
            if len(pc) == 0:
                continue
            if Mode ==0:
                color = [0, 0, 0]
            else:
                color = COLOR[point_cloud[i]['class']]
            Range, Angle, z = Cathesian_to_RA(pc)
            if not Save_Lines:
                plt.scatter(Angle, Range, c=np.array(color).reshape(1, -1) / 255, s=6)



    # Adding the legend for detected classes
    if not Save_Lines:
        if Mode == 1:
            patches = [mpatches.Patch(color=np.array(COLOR[idx_class]).reshape(1, -1) / 255, label=CLASSES[idx_class])
                       for idx_class in detected_classes]
            plt.legend(handles=patches, loc='upper right')
        else:
            Patch = [mpatches.Patch(color='k', label='Obstacles'), mpatches.Patch(color='lightcoral', label='Open Space'),mpatches.Patch(color='lightgrey', label='Unknown')]
            plt.legend(handles=Patch, loc='upper right')

        plt.xlim(-Radar_FOV/2, Radar_FOV/2)
        plt.ylim(0, Radar_Range)
        plt.xlabel('Angle (°)')
        plt.ylabel('Range (m)')
        plt.title('DBSCAN Clustering Range-Angle')

        if Mode == 0:
            return Angle_step_2,Range_step
        else:
            return None
    else:
        return Lines_to_save

# Function to plot the Cartesian scatter plot
def plot_cartesian_scatter(point_cloud,Mode,Road_Plan_Equation,Returned_data):

    # Generate the outside of FOV limit
    theta = np.radians(np.linspace(-Radar_FOV / 2, Radar_FOV / 2, 100))
    x_fov = Radar_Range * np.cos(theta)
    y_fov = Radar_Range * np.sin(theta)

    # Add the origin at the beginning and end of the FOV
    x_fov = np.concatenate([[0], x_fov, [0]])
    y_fov = np.concatenate([[0], y_fov, [0]])

    # Set the background color to grey using the current axis
    plt.gca().set_facecolor('dimgray')

    # Fill the area inside the FOV in white
    plt.fill(-y_fov, x_fov, 'lightgrey')

    if Mode == 0:
        Angle,Range=Returned_data
        # Convert to Cartesian
        x = Range * np.sin(np.radians(Angle))
        y = Range * np.cos(np.radians(Angle))
        # fil inside in red
        plt.fill(x, y, c='lightcoral')

    for i in range(len(point_cloud)):
        if point_cloud[i]['class'] == 16 :
            continue
        pc = point_cloud[i]['point_cloud']
        pc = remove_out_of_drivable_space(pc,Road_Plan_Equation,Limit_vehicle)
        if Mode  == 0:
            color = [0, 0, 0]
        else:
            color = COLOR[point_cloud[i]['class']]
        plt.scatter(-pc[:, 1], pc[:, 0], c=np.array(color).reshape(1, -1) / 255, s=4)

    # Add Legend:
    Patch = [mpatches.Patch(color='dimgray', label='Outside of Radar FOV')]
    plt.legend(handles=Patch, loc='upper right')

    plt.xlim(-Radar_Range*np.sin(np.radians(Radar_FOV/2)), Radar_Range*np.sin(np.radians(Radar_FOV/2)))
    plt.ylim(0, Radar_Range)
    # aspect equal
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('DBSCAN Clustering Cartesian')

    if Mode == 0:
        return x,y
    else:
        return None

# -----


def Cathesian_to_RA(point_Cloud):

    x = point_Cloud[:, 0]
    y = -point_Cloud[:, 1]
    z = point_Cloud[:, 2]

    #Range:
    r = np.sqrt(x**2 + y**2 + z**2)
    angle = np.arctan2(y, x)
    angle = np.degrees(angle)
    return r, angle,z

def RA_to_Cartesian(r, angle, z):
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle)

    # Calculate x and y from r and angle
    x = r * np.cos(angle_rad)
    y = r * np.sin(angle_rad)

    # Return the Cartesian coordinates
    return x, -y, z



def get_files(path):
    if path is not None:
        if os.path.isdir(path):
            files = natsorted(os.listdir(path))
            return [os.path.join(path, file) for file in files]
        else:
            return [path]
    return []

def visualize_data(args):

    if args.sequence is None:
        # Select a random sequence
        Sequence = random.choice(os.listdir(args.Data_path))
        args.sequence = Sequence

    Sequence_path = os.path.join(args.Data_path, args.sequence)

    if args.GT_version ==0:
        GT_version='GT'
    else:
        GT_version='GT_2'

    if args.frame_number is not None:
        # Radar
        Radar_data_path = os.path.join(Sequence_path, 'Radar_Data')
        Radar_files = os.listdir(Radar_data_path)
        Radar_files = natsorted(Radar_files)
        Radar_data_path += '/' + Radar_files[args.frame_number]
        # GT
        GT_path = os.path.join(Sequence_path, GT_version) + '/Frame' + str(args.frame_number) + '.npy'
        # Frontal Image
        Frontal_Image_path = os.path.join(Sequence_path, 'Resized_images') + '/' + str(args.frame_number) + '.png'
    else:
        # Radar
        Radar_data_path = os.path.join(Sequence_path, 'Radar_Data')
        # GT
        GT_path = os.path.join(Sequence_path, GT_version)
        # Frontal Image
        Frontal_Image_path = os.path.join(Sequence_path, 'Resized_images')

    Sequence=os.path.basename(Sequence_path)

    if args.mode_open_space in [0,1]:

        Mode = args.mode_open_space # 0: Do_Open_space, 1: Do_DBSCAN

        Sequence_path = Dataset_path + Sequence + '/Sorted_Point_Cloud/'

        plt.figure(figsize=(17, 12))
        for i in range(len(os.listdir(Sequence_path)) - 1):
            print('Frame:', i)
            # Data Path:
            Data_path = Sequence_path +'/'+ str(i) + '.npy'

            Point_Cloud = np.load(Data_path, allow_pickle=True)

            # Visualisation mode:
            if not Save_Lines:
                # Find_image:
                Image_path = Data_path.replace('Sorted_Point_Cloud', 'Resized_images')
                Image_path = Image_path.replace('npy', 'png')
                Image_dirname = os.path.dirname(Image_path)
                Image_path = os.path.join(Image_dirname, str(i + 1) + '.png')

                # Find_Road_equation:
                Road_Plan_Equation_path = Data_path.replace('Sorted_Point_Cloud', 'Road_Plan_Equation')
                Road_Plan_Equation = np.load(Road_Plan_Equation_path, allow_pickle=True)

                # Original Image:
                Original_image = np.zeros((500, 2000, 4))
                Original_image[:, 420:1590, :] = plt.imread(Image_path)
                Original_image = Original_image[:, :, :3]

                plt.subplot(2, 1, 1)
                plt.imshow(plt.imread(Image_path))
                plt.xlim(0, 1170)
                plt.ylim(500, 0)
                plt.axis('off')
                plt.title('Front camera:')

                plt.subplot(2, 2, 3)
                Returned_data = plot_range_angle_scatter(Point_Cloud, Mode, Save_Lines, Road_Plan_Equation, i,Sequence)

                plt.subplot(2, 2, 4)
                Returned_data = plot_cartesian_scatter(Point_Cloud, Mode, Road_Plan_Equation, Returned_data)

                if Mode == 2:
                    y, x = Returned_data
                    y = -y

                    pts2d=project_points(x, y,Road_Plan_Equation)

                    plt.subplot(2, 1, 1)
                    plt.imshow(plt.imread(Image_path))
                    plt.plot(pts2d[:, 0], pts2d[:, 1], c='b', linewidth=2)
                    plt.xlim(0, 1170)
                    plt.ylim(500, 0)
                    plt.axis('off')
                    Patch = [mpatches.Patch(color='b', label='Open Space Limit')]
                    plt.legend(handles=Patch, loc='upper right')
                    plt.title('Front camera:')

                    # Figure title:
                    plt.suptitle('Sequence:' + Sequence + ' Frame:' + str(i), fontsize=20)

                plt.pause(0.1)
                plt.tight_layout()
                print('yes')
                plt.clf()


            else:
                Lines_to_save = plot_range_angle_scatter(Point_Cloud, Mode, Save_Lines)
                np.save(Dataset_path + Sequence + '/Lines/' + str(i) + '.npy', Lines_to_save)

    else:

        if Radar_data_path is None and Frontal_Image_path is None and GT_path is None:
            print('No data to visualize')
            return

        Radar_File = get_files(Radar_data_path)
        Frontal_Image = get_files(Frontal_Image_path)
        GT_File = get_files(GT_path)
        Lines_GT=get_files(os.path.dirname(GT_path)+"/Lines")
        Point_clouds=get_files(os.path.dirname(GT_path)+"/Sorted_Point_Cloud")
        Road_Plan_Equation = get_files(os.path.dirname(GT_path)+"/Road_Plan_Equation")

        if args.model_pred:
            Pred_Full,_,_=Test(args)

        plt.figure(figsize=(10, 8))

        for frame in range(len(GT_File)-1):
            if Frontal_Image and Radar_File:
                # Both camera and radar are available
                plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st plot
                image = plt.imread(Frontal_Image[frame+1])
                plt.imshow(image)
                plt.title('Frontal Camera Image')
                # Remove axis
                plt.axis('off')
                plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd plot
                Radar_data = Load_Radar_Data(Radar_File[frame], cfg,Mode_Visualisation=True)
                if args.GT_point_cloud:
                    plt.imshow(Radar_data, extent=[-cfg.FOV/2, cfg.FOV/2, 0, cfg.Radar_Range], aspect='equal')
                    plt.title('Radar Data')
                    GT = np.load(GT_File[frame])
                    X=GT[:, 0]
                    Y=GT[:, 1]

                    plt.scatter(X, Y, c='r', s=4)
                    plt.xlim(-cfg.FOV/2, cfg.FOV/2)
                    plt.ylim(0, cfg.Radar_Range)

                else:
                    gt_data=Load_GT_Data(GT_File[frame], cfg)

                    radar_data, gt_data=adjust_radar_data_based_on_range_and_gt([Radar_data], [gt_data], cfg)
                    plt.imshow(radar_data[0], extent=[-cfg.FOV/2, cfg.FOV/2, 0, cfg.Radar_Range], aspect='equal')
                    plt.title('Radar Data')

                    GT=gt_data[0][:,1]

                    Points=[]
                    if args.model_pred:
                        Pred_Points=[]
                    for i in range(cfg.GT_Output_shape[0]):
                        Points.append([i * cfg.FOV / (cfg.GT_Output_shape[0]) - cfg.FOV/2, GT[i]*cfg.Radar_Range/cfg.Output_vertices])
                        Points.append([(i + 1) * cfg.FOV / (cfg.GT_Output_shape[0]) - cfg.FOV/2, GT[i]*cfg.Radar_Range/cfg.Output_vertices])
                        # plt.plot([i * cfg.FOV / (cfg.GT_Output_shape[0] + 1) - cfg.FOV/2, (i + 1) * cfg.FOV / (cfg.GT_Output_shape[0] + 1) - cfg.FOV/2],[GT[i]*cfg.Radar_Range/cfg.Output_vertices, GT[i]*cfg.Radar_Range/cfg.Output_vertices], c='r')
                        if args.model_pred:
                            Pred_Points.append([i * cfg.FOV / (cfg.GT_Output_shape[0]) - cfg.FOV/2, Pred_Full[frame][i][0]*cfg.Radar_Range/cfg.Output_vertices])
                            Pred_Points.append([(i + 1) * cfg.FOV / (cfg.GT_Output_shape[0]) - cfg.FOV/2, Pred_Full[frame][i][0]*cfg.Radar_Range/cfg.Output_vertices])

                    plt.plot(np.array(Points)[:,0],np.array(Points)[:,1], c='r')
                    if args.model_pred:
                        # Add a point at the end and at the begening to close the polygon
                        Pred_Points.append([cfg.FOV/2, 0])
                        Pred_Points.insert(0, [-cfg.FOV/2, 0])
                        plt.fill(np.array(Pred_Points)[:,0], np.array(Pred_Points)[:,1], color='g', alpha=0.5)

                    # Get 3D points:
                    # Load 3d:
                    Road_Equation=np.load(Road_Plan_Equation[frame],allow_pickle=True)

                    if args.model_pred:
                        Angle_pred, Range_pred = np.array(Pred_Points)[:, :2].T
                        x_pred, y_pred, _ = RA_to_Cartesian(Range_pred, Angle_pred, None)
                        pts2d_pred = project_points(x_pred, y_pred, Road_Equation)

                    Angle, Range = np.array(Points)[:, :2].T
                    x,y,_=RA_to_Cartesian(Range,Angle, None)
                    pts2d=project_points(x,y,Road_Equation)

                    plt.subplot(2, 1, 1)
                    plt.imshow(image)
                    plt.plot(pts2d[:, 0], pts2d[:, 1], c='r', linewidth=2)
                    plt.fill(pts2d_pred[:, 0], pts2d_pred[:, 1], c='g', alpha=0.5)
                    # X,Y limits
                    plt.xlim(0, 1170)
                    plt.ylim(500, 0)
                    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd plot



                plt.xlim(-cfg.FOV/2, cfg.FOV/2)
                plt.ylim(0, cfg.Radar_Range)
                if args.model_pred:
                    plt.legend(['Drivable Area limit','Prediction'], loc='upper right')
                else:
                    plt.legend(['Drivable Area limit'], loc='upper right')
                plt.xlabel('Angle (°)')
                plt.ylabel('Range (m)')

                #plt.tight_layout()

            elif Frontal_Image:
                print('yes')
                # Only the camera image is available
                image = plt.imread(Frontal_Image[i])
                plt.plot(pts2d[:, 0], pts2d[:, 1], c='b', linewidth=2)
                plt.imshow(image)
                plt.title('Frontal Camera Image')
                plt.axis('off')
            elif Radar_File:
                # Only the radar data is available
                Radar_data = Load_Radar_Data(Radar_File[i], cfg)
                plt.imshow(Radar_data, extent=[-60, 60, 0, 50])
                plt.title('Radar Data')
                plt.legend(['Drivable Area limit'], loc='upper right')
                plt.xlabel('Angle (°)')
                plt.ylabel('Range (m)')

            plt.pause(1/args.fps)
            #plt.savefig('/home/watercooledmt/PycharmProjects/Radar_Open_Space_Segmentation/Test/Test_pred/Frame'+str(frame)+'.png')
            plt.clf()
