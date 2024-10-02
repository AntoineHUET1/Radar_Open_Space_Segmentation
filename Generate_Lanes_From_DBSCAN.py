import os
from tracemalloc import Frame

import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import Range

from ROSS.Utils.Visualize import Cathesian_to_RA
from ROSS.Utils.Generate_Graph import COLOR,CLASSES,Transform_matrix,image_width,image_height,FOV_width,FOV_height
import matplotlib.patches as mpatches
import open3d as o3d

# Parameters:
Angular_rez=1
Angular_rez_scan=0.1

Limit_vehicle= -0.87 + 1.69 + 0.5 # Road limit + Vehicle height + Margin

# Lines:
Do_Lines=False
Mode = 2 # 0: Do_Lines, 1: Do_DBSCAN, 2: Do_Open_space

Number_of_lines=5
angle_threshold=0.5

# Radar Spec:
Radar_FOV = 120
Radar_Range = 50

# Data Visualisation:
Show_DBSCAN = False
Show_Road = False
Save_Lines=False # Remove all the plots and save the lines

# Point Cloud Modification:
Remove_Out_of_Drivable_Space = True

# Functions:

def Remove_small_gap(Angle_step_2,Range_step,Angular_rez_max_gap=4):
    for i in range(len(Angle_step_2)-1):
        if Range_step[i] == Radar_Range and Range_step[i+1] == Radar_Range:
            # Estimate distance
            Angular_rez=Angle_step_2[i+1]-Angle_step_2[i]
            if Angular_rez<Angular_rez_max_gap:
                if i>0:
                    Range_step[i]=Range_step[i-1]
                    Angle_step_2[i]=Angle_step_2[i]+Angular_rez/2
                if i<len(Angle_step_2)-2:
                    Range_step[i+1]=Range_step[i+2]
                    Angle_step_2[i+1]=Angle_step_2[i+1]-Angular_rez/2
    return Angle_step_2,Range_step

    plt.scatter(Angle_step_2,Range_step)
    plt.show()
def Remove_Out_of_Drivable_Space(Point_Cloud,Road_Plan_Equation):
    # Remove the points that are out of the drivable space

    a=Road_Plan_Equation[0]['a']
    b=Road_Plan_Equation[0]['b']
    c=Road_Plan_Equation[0]['c']
    z_road = a*Point_Cloud[:,0]+b*Point_Cloud[:,1]+c

    z=Point_Cloud[:,2]
    # Height of the point above the road
    Height = z - z_road

    keep = np.where(Height < Limit_vehicle)[0]
    return Point_Cloud[keep]

def Split_pc(pc,Max_delta_angle=6):

    Range, Angle, _ = Cathesian_to_RA(pc)

    # Sort the points by angle
    idx = np.argsort(Angle)
    Angle = Angle[idx]
    pc = pc[idx]

    # Split the point cloud if the angle variation is too big
    Split_idx=[0]
    for i in range(len(Angle)-1):
        if Angle[i+1]-Angle[i]>Max_delta_angle:

            Split_idx.append(i+1)

    if len(Split_idx)>1:
        Split_idx.append(len(Angle))
        Split_pc=[]
        for i in range(len(Split_idx)-1):
            Split_pc.append(pc[Split_idx[i]:Split_idx[i+1]])
        return Split_pc
    else:
        return [pc]


# Function to plot the frontal camera image
def plot_frontal_image(image_path):
    plt.imshow(plt.imread(image_path))
    plt.title('Frontal Camera Image')
    plt.axis('off')

def map_points(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform on 3x1 point(s)

    Args:
        m: a 4x4 transform
        v: Nx3 point matrix
    """
    return (np.dot(m[:3, :3], v.T) + m[:3, 3, None]).T

def remove_outliers(lines, angle_threshold):
    # Step 1: Concatenate all angles from the lines
    all_angles = np.concatenate([line[:, 0] for line in lines])
    min_angle = np.min(all_angles)
    max_angle = np.max(all_angles)

    # Step 2: Generate angles in the range with angle_threshold step
    angle_steps = np.arange(min_angle, max_angle, angle_threshold)
    # Check if max_angle is included, and append if necessary
    if max_angle not in angle_steps:
        angle_steps = np.append(angle_steps, max_angle)

    # Step 3: Interpolate range values for each angle in angle_steps
    interpolated_ranges = []
    for angle in angle_steps:
        ranges_at_angle = []

        # For each line, find the interpolated range at the current angle
        for line in lines:
            # Use np.interp to find the range for the specific angle
            if min(line[:, 0]) <= angle <= max(line[:, 0]):  # Ensure angle is within the line's range
                interpolated_range = np.interp(angle, line[:, 0], line[:, 1])
                ranges_at_angle.append(interpolated_range)

        # Step 4: Filter out outliers using the previous method
        if len(ranges_at_angle) > 0:
            mean_range = np.mean(ranges_at_angle)
            std_dev_range = np.std(ranges_at_angle)
            threshold = 1  # Using 2 standard deviations as threshold
            filtered_ranges = [r for r in ranges_at_angle if abs(r - mean_range) <= threshold * std_dev_range]

            # Step 5: Save the angle with the average of the filtered range values
            if len(filtered_ranges) > 0:
                avg_range = np.mean(filtered_ranges)
                interpolated_ranges.append([angle, avg_range])

    # Step 6: Print the final result with angles and their average ranges
    interpolated_ranges = np.array(interpolated_ranges)

    return interpolated_ranges

# Function to plot the range-angle scatter plot
def plot_range_angle_scatter(point_cloud, show_road,Mode,Save_Lines,Road_Plan_Equation,frame):
    Lines_to_save=[]
    detected_classes = []
    for i in range(len(point_cloud)):
        if point_cloud[i]['class'] == 16 and not show_road:
            continue
        pc = point_cloud[i]['point_cloud']
        pc = Remove_Out_of_Drivable_Space(pc,Road_Plan_Equation)
        if len(pc) == 0:
            continue
        if point_cloud[i]['class'] == 37:
            List_pc = Split_pc(pc)
        else:
            List_pc = [pc]

        for pc in List_pc:

            if Mode in [0, 2]:
                color = [0, 0, 0]
            else:
                color = COLOR[point_cloud[i]['class']]
            Range, Angle, z = Cathesian_to_RA(pc)
            if not Save_Lines:
                plt.scatter(Angle,Range, c=np.array(color).reshape(1, -1) / 255, s=6)
            if point_cloud[i]['class'] not in detected_classes:
                detected_classes.append(point_cloud[i]['class'])

            if Mode in [0, 2]:

                Z_max= np.max(z)
                Z_min= np.min(z)
                Z_min= Z_min + 0.1*(Z_max-Z_min)
                # Remove first  10% of the data

                plt.scatter(Angle, Range, c=np.array(color).reshape(1, -1) / 255, s=6)
                # Split Z into 10 equal parts
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
                        #if not Save_Lines and Mode==0:
                        #plt.plot(Line[:, 0], Line[:, 1], c='b')
                        Lines.append(Line)
                if len(Lines)>0:
                    Output_Lines=remove_outliers(Lines,angle_threshold)
                    Lines_to_save.append({'class':point_cloud[i]['class'],'lines':Output_Lines})
                    if not Save_Lines:
                        if Mode == 0:
                            plt.plot(Output_Lines[:, 0], Output_Lines[:, 1], c='b')

    
    if Mode ==2:
        Angle_step=[line_data['lines'] for line_data in Lines_to_save]
        # Flatten: `
        Angle_step = np.concatenate(Angle_step)
        Angle_step_2 =[]
        # Sort:
        Angle_step = Angle_step[np.argsort(Angle_step[:, 0])]
        Angle_step = Angle_step[:, 0]

        Range_step=[]
        Line_idx=[]

        if Radar_FOV/2 not in Angle_step:
            Angle_step = np.append(Angle_step, Radar_FOV/2)
        if -Radar_FOV/2 not in Angle_step:
            Angle_step = np.insert(Angle_step, 0, -Radar_FOV/2)
        #plt.figure()
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
            if len(ranges_at_angle) > 0:
                # Find the minimum index
                min_index_line = np.argmin(ranges_at_angle)
                Line_idx.append(line_idx[min_index_line])
                Range_step.append(ranges_at_angle[min_index_line])
                Angle_step_2.append(angle)

                # if previou
                # Check if the line as changed
                if len(Line_idx)>1:
                    if Line_idx[-1]!=Line_idx[-2]:
                        #plt.scatter(Angle_step_2[-1],Range_step[-1],c='r')
                        #plt.scatter(Angle_step_2[-2],Range_step[-2],c='g')
                        #plt.text(Angle_step_2[-2],Range_step[-2],str(Line_idx[-2]))
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


        keep=np.where(np.array(Range_step)<Radar_Range)[0]
        #Stack Angle and Range in GT array:
        GT=np.vstack((np.array(Angle_step_2)[keep],np.array(Range_step)[keep])).T
        #save GT
        np.save('/home/watercooledmt/PycharmProjects/Radar_Open_Space_Segmentation/data/ROSS_Dataset/'+Sequence+'/GT_2/'+str(frame)+'.npy',GT)


        #breakpoint()
        Angle_step_2,Range_step=Remove_small_gap(Angle_step_2,Range_step)

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
            if point_cloud[i]['class'] == 16 and not show_road:
                continue
            pc = point_cloud[i]['point_cloud']
            pc = Remove_Out_of_Drivable_Space(pc, Road_Plan_Equation)
            if len(pc) == 0:
                continue
            if Mode in [0, 2]:
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

        if Mode == 2:
            return Angle_step_2,Range_step
        else:
            return None
    else:
        return Lines_to_save


# Function to plot the Cartesian scatter plot
def plot_cartesian_scatter(point_cloud, show_road,Mode,Road_Plan_Equation,Returned_data):

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

    if Mode == 2:
        Angle,Range=Returned_data
        # Convert to Cartesian
        x = Range * np.sin(np.radians(Angle))
        y = Range * np.cos(np.radians(Angle))
        # fil inside in red
        plt.fill(x, y, c='lightcoral')

    for i in range(len(point_cloud)):
        if point_cloud[i]['class'] == 16 and not show_road:
            continue
        pc = point_cloud[i]['point_cloud']
        pc = Remove_Out_of_Drivable_Space(pc,Road_Plan_Equation)
        if Mode in [0, 2]:
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

    if Mode == 2:
        return x,y
    else:
        return None

plt.figure(figsize=(17, 12))
for Sequence in (os.listdir('/home/watercooledmt/PycharmProjects/Radar_Open_Space_Segmentation/data/ROSS_Dataset/')):
    #Sequence='20200616_145121_part7_2575_2860'
    print('Sequence:',Sequence)
    # Create a saving folder if not exist:
    if not os.path.exists('/home/watercooledmt/PycharmProjects/Radar_Open_Space_Segmentation/data/ROSS_Dataset/' + Sequence + '/Lines/'):
        os.makedirs('/home/watercooledmt/PycharmProjects/Radar_Open_Space_Segmentation/data/ROSS_Dataset/' + Sequence + '/Lines/')

    if not os.path.exists('/home/watercooledmt/PycharmProjects/Radar_Open_Space_Segmentation/data/ROSS_Dataset/' + Sequence + '/Visualization/'):
        os.makedirs('/home/watercooledmt/PycharmProjects/Radar_Open_Space_Segmentation/data/ROSS_Dataset/' + Sequence + '/Visualization/')

    if not os.path.exists('/home/watercooledmt/PycharmProjects/Radar_Open_Space_Segmentation/data/ROSS_Dataset/' + Sequence + '/GT_2/'):
        os.makedirs('/home/watercooledmt/PycharmProjects/Radar_Open_Space_Segmentation/data/ROSS_Dataset/' + Sequence + '/GT_2/')
    #else:
        #continue

    Sequence_path='/home/watercooledmt/PycharmProjects/Radar_Open_Space_Segmentation/data/ROSS_Dataset/'+Sequence+'/Sorted_Point_Cloud/'

    for i in range(len(os.listdir(Sequence_path))):
        print('Frame:',i)
        # Data Path:
        Data_path = Sequence_path+str(i)+'.npy'

        Point_Cloud = np.load(Data_path, allow_pickle=True)

        if not Save_Lines:
            # Find_image:
            Image_path = Data_path.replace('Sorted_Point_Cloud', 'Resized_images')
            Image_path = Image_path.replace('npy', 'png')

            # Find_Road_equation:
            Road_Plan_Equation_path = Data_path.replace('Sorted_Point_Cloud', 'Road_Plan_Equation')
            Road_Plan_Equation = np.load(Road_Plan_Equation_path,allow_pickle=True)

            # Original Image:
            Original_image=np.zeros((500, 2000, 4))
            Original_image[:, 420:1590, :] = plt.imread(Image_path)
            Original_image= Original_image[:, :, :3]

            plt.subplot(2, 1, 1)
            plt.imshow(plt.imread(Image_path))
            plt.xlim(0, 1170)
            plt.ylim(500, 0)
            plt.axis('off')
            plt.title('Front camera:')

            plt.subplot(2, 2, 3)
            Returned_data=plot_range_angle_scatter(Point_Cloud, Show_Road, Mode, Save_Lines, Road_Plan_Equation,i)

            plt.subplot(2, 2, 4)
            Returned_data=plot_cartesian_scatter(Point_Cloud, Show_Road,Mode,Road_Plan_Equation,Returned_data)

            if Mode==2:
                y,x=Returned_data
                y=-y

                z=Road_Plan_Equation[0]['a']*x+Road_Plan_Equation[0]['b']*y+Road_Plan_Equation[0]['c']
                point_cloud_in_camera_ref = map_points(Transform_matrix, np.transpose(np.array([x,y,z])))

                # project points inside the image:
                pts2 = point_cloud_in_camera_ref.T
                azimut = np.arctan2(pts2[0, :], pts2[2, :])
                norm_xz = np.linalg.norm(pts2[[0, 2], :], axis=0)
                elevation = np.arctan2(pts2[1, :], norm_xz)
                x = np.round(image_width / 2 + azimut * (image_width / FOV_width))
                y = np.round(image_height / 2 + elevation * (image_height / FOV_height))
                pts2d = np.column_stack((x, y))
                # Change last and first point to close the polygon
                pts2d[0, :] = [420,500]
                pts2d[-1, :] = [1590,500]

                # Resize the points to the original image size
                pts2d[:, 0] = pts2d[:, 0] -420

                plt.subplot(2, 1, 1)
                # plot_frontal_image(Image_path)
                plt.imshow(plt.imread(Image_path))
                plt.plot(pts2d[:, 0], pts2d[:, 1], c='b', linewidth=2)
                plt.xlim(0, 1170)
                plt.ylim(500, 0)
                plt.axis('off')
                Patch = [mpatches.Patch(color='b', label='Open Space Limit')]
                plt.legend(handles=Patch, loc='upper right')
                plt.title('Front camera:')

                # Figure title:
                plt.suptitle('Sequence:'+Sequence+' Frame:'+str(i), fontsize=20)


            # figure tight layout
            plt.tight_layout()

            #plt.pause(0.1)
            #plt.show()
            plt.savefig('/home/watercooledmt/PycharmProjects/Radar_Open_Space_Segmentation/data/ROSS_Dataset/' + Sequence + '/Visualization/' + str(i) + '.jpg')
            plt.clf()

        else:
            Lines_to_save=plot_range_angle_scatter(Point_Cloud, Show_Road, Mode,Save_Lines)
            np.save('/home/watercooledmt/PycharmProjects/Radar_Open_Space_Segmentation/data/ROSS_Dataset/' + Sequence + '/Lines/' + str(i) + '.npy',Lines_to_save)




'''
if Show_DBSCAN:

    pcd_classified = []
    for i in range(len(Point_Cloud)):

        Color = COLOR[Point_Cloud[i]['class']]
        Lidar_points = Point_Cloud[i]['point_cloud']

        pcd = o3d.geometry.PointCloud()
        # 3D points:
        pcd.points = o3d.utility.Vector3dVector(Lidar_points)
        pcd.paint_uniform_color([x / 255.0 for x in Color])

        pcd_classified.append(pcd)


    o3d.visualization.draw_geometries(pcd_classified)

plt.figure(figsize=(10, 10))
plt.subplot(2, 1,1)
plt.imshow(plt.imread(Image_path))
plt.title('Frontal Camera Image')
plt.axis('off')

plt.subplot(2, 2, 3)
Detected_Classes=[]
for i in range(len(Point_Cloud)):
    if Point_Cloud[i]['class'] == 16 and not Show_Road:
        continue
    point_cloud = Point_Cloud[i]['point_cloud']
    Color = COLOR[Point_Cloud[i]['class']]

    Class = CLASSES[Point_Cloud[i]['class']]
    Range, Angle, z = Cathesian_to_RA(point_cloud)
    plt.scatter(Angle, Range, c=np.array(Color).reshape(1, -1) / 255, s=4)
    if Point_Cloud[i]['class'] not in Detected_Classes:
        Detected_Classes.append(Point_Cloud[i]['class'])
Patches=[]
for idx_class in Detected_Classes:
    Patches.append(mpatches.Patch(color=np.array(COLOR[idx_class]).reshape(1, -1) / 255, label=CLASSES[idx_class]))

plt.legend(handles=Patches, loc='upper right')

plt.xlim(-60, 60)
plt.ylim(0, 50)
plt.xlabel('Angle (°)')
plt.ylabel('Range (m)')
plt.title('DBSCAN Clustering Range-Angle')

plt.subplot(2, 2, 4)
for i in range(len(Point_Cloud)):
    if Point_Cloud[i]['class'] == 16 and not Show_Road:
        continue
    point_cloud = Point_Cloud[i]['point_cloud']
    Color = COLOR[Point_Cloud[i]['class']]

    Class = CLASSES[Point_Cloud[i]['class']]
    plt.scatter(-point_cloud[:,1],point_cloud[:,0], c=np.array(Color).reshape(1, -1) / 255, s=4)

plt.xlim(-50, 50)
plt.ylim(0, 50)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('DBSCAN Clustering Cartesian')
plt.show()

for i in range(len(Point_Cloud)):
    point_cloud = Point_Cloud[i]['point_cloud']
    Color = COLOR[Point_Cloud[i]['class']]
    Range, Angle, z = Cathesian_to_RA(point_cloud)

    Line = [[Angle[np.argmin(Angle)], Range[np.argmin(Angle)]]]
    for j in np.arange(min(Angle), max(Angle), Angular_rez):
        keep = np.where((Angle >= j) & (Angle < j + Angular_rez))[0]
        if len(keep) > 0:
            min_index = np.argmin(Range[keep])
            Line.append([Angle[keep[min_index]], Range[keep[min_index]]])

    Line.append([Angle[np.argmax(Angle)], Range[np.argmax(Angle)]])
    Line = np.array(Line)
'''
