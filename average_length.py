"""
@author: Tijmen van Wel

Last edited on Jan 10 2025

@editor: Marjolein van Tol
"""

from skimage.morphology import skeletonize, medial_axis, square, binary_opening
import numpy as np
import math
import cv2
import time
from fitter import Fitter
import csv
import os
from scipy import stats



#--------------Parameters------------------------------------------------------------------
#--------------Generic---------------
global pixel_to_microm
# #--------------Distance_calc---------
global min_length_skeleton    
global list_width
global deg_dif
global check_distance

visualize = True
skel_colour_values = [255, 255, 255]
skeletonized = False
# #--------------Fourier---------------
global upper_s_len
global lower_s_len
global full_coverage
global lines_from_center
global interval_for_full_coverage

#------------------------------------------------------------------------------------------

#Check if coordinate is inside the picture
def in_bounds(main_array, x: int, y: int) -> bool:
    if len(main_array) > y and len(main_array[0]) > x:
        return True
    else:
        return False

#Find neighbours of coordinate
def neighbours(main_array, current_x: int, current_y: int, colour: list[int]) -> list[list[int]]: 
    current_found_neighbours=[]
    end_x = current_x + 1
    end_y = current_y + 1
    i = current_y - 1 
    while i <= end_y and in_bounds(main_array, 0, i):
        j = current_x - 1
        while j <= end_x and in_bounds(main_array, j, i):
            if list(main_array[i][j]) == colour: #skimage creates green skeleton. If other way of skeletonization is used, change this RGB value
                current_found_neighbours.append([j,i])
            j += 1
        i += 1
    if [current_x, current_y] in current_found_neighbours:
        current_found_neighbours.remove([current_x,current_y])
    return current_found_neighbours

#checks if only one neighbour
def single_neighbour(list_of_neighbours: list) -> bool:
    if len(list_of_neighbours) == 1:
        return True
    else:
        return False

#changes input pixel to black
def remove_checked_pixels(main_array, current_x, current_y): #currently unused
    i = 0
    while i < 3:
        main_array[current_y][current_x][i] = 0
        i += 1
    return main_array

#Finds furthest connected point 
def furthest_point(main_array, current_x: int, current_y: int, colour: list[int]) -> tuple:
    neigh = neighbours(main_array, current_x, current_y, colour)
    if len(neigh) == 0:
        return (current_x, current_y)
    else:
        k = 0
        while k < 3:
            main_array[current_y][current_x][k] = 0
            k += 1

        return furthest_point(main_array, neigh[0][0], neigh[0][1], colour)

#Finds angle given 3 lengths of triangle
def angle(a, b, c) -> float: #credits naar Linde Schoenmaker Myobibril_trace_v21
    try:
        ang = math.degrees(math.acos((c ** 2 - b ** 2 - a ** 2) / (-2.0 * a * b)))
    except ValueError:
        ang = 180

        if abs(c - a - b) > 1e-10:
            print("Warning: possible problem calculating the angle")
            print(f"c - a - b = {c - a - b}")

    return ang

#Finds angle of line compared to the horizontal
def ang(length: float, point_1: tuple[int], point_2: tuple[int]) -> float:
    a = abs(point_1[1] - point_2[1])
    b = length
    c = abs(point_1[0] - point_2[0])
    if a == 0:
        return 0 
    return angle(a, b, c)

#Finds slope of line, aka the a value in the formula y = ax + b
def slope(point_1: tuple[int], point_2: tuple[int]) -> float:
    if point_1[0] - point_2[0] == 0:
        return 'vertical'
    else:
        a = (point_1[1] - point_2[1])/(point_1[0]-point_2[0])
        return a

#Calculates the b value in the formula y = ax + b for a line
def calc_line_formula_b(slope: float, x: float, y: float) -> float:
    return y - (slope * x)

#Calculate the intersec point of a line and another lines perpendicular when the two original lines are parrallel 
def calc_intersec_point_parallel(slope: float, slope_perp: float, x: float, y: float) -> tuple:
    b = calc_line_formula_b(slope, x, y)
    intersec_x = b/(slope_perp - slope)
    return (intersec_x, slope_perp * intersec_x )

#Converts the formula y = ax + b to x = cy + d 
def convert_to_y(a: float, b: float) -> tuple[float, float]:
    y_a = 1/a
    y_b = (-b)/a
    return y_a, y_b

#Checks whether a further point is in bounds and changes it if not
def calc_further_point(a: float, b: float, known_cord: float, main_array, x_or_y='x') -> int:
    new_cord = round((known_cord * a)+b)
    if x_or_y == 'x':
        max_len = len(main_array[0])
    else:
        max_len = len(main_array)
    if max_len > new_cord:
        return new_cord
    else: 
        return max_len - 1

#Sorts points based on their x or y coordinates, can be used for one line or two lines
def sort_points(point_1: tuple, point_2: tuple, x_or_y = 'x') -> tuple:
    if x_or_y == 'x':
        index = 0
    else:
        index = 1
    if type(point_1[0]) == int:
        if point_1[index] > point_2[index]:
            return (point_2, point_1)
        else:
            return (point_1, point_2)
    elif type(point_1[0][0]) == int:
        if point_1[0][index] > point_2[0][index]:
            return (point_2, point_1)
        else:
            return (point_1, point_2)

#Calculates the intersection point of the two input line
def calc_intersec(line_1: tuple, line_2: tuple) -> tuple[int]:
    a_1 = slope(line_1[0], line_1[1])
    a_2 = slope(line_2[0], line_2[1])
    if a_1 == a_2:
        return (0, 0) #gets filtered out later on, no intersection when lines are parallel
    if a_1 == 'vertical': #vertical lines don't have an y = ax + b equation
        sorts = sort_points(line_2[0], line_2[1])
        if line_1[0][0] > sorts[0][0] and line_1[0][0] < sorts[1][0]: #checks whether intersection is on line
            b = calc_line_formula_b(a_2, line_2[0][0], line_2[0][1])
            intersec_y = (line_1[0][0]*a_2) + b
            return (line_1[0][0], intersec_y)
        else:
            return (0,0) #gets filtered out later on
    if a_2 == 'vertical': #vertical lines don't have an y = ax + b equation
        sorts = sort_points(line_2[0], line_2[1], 'y')
        b = calc_line_formula_b(a_1, line_1[0][0], line_1[0][1])
        intersec_y = line_2[0][0]*a_1 + b
        if intersec_y > sorts[0][1] and intersec_y < sorts[1][1]: #checks whether intersection is on line
            return (line_2[0][0], intersec_y)
        else:
            return (0,0) #gets filtered out later on
    b_1 = calc_line_formula_b(a_1, line_1[0][0], line_1[0][1])
    b_2 = calc_line_formula_b(a_2, line_2[0][0], line_2[0][1])
    intersec_x = (b_1 - b_2)/(a_2 - a_1)
    intersec_y = (intersec_x*a_1) + b_1
    return (intersec_x, intersec_y)

def check_if_cross(line_1: tuple, line_2: tuple, check_dist: int, main_array, deg_dif: int = 20, draw: bool = False) -> list[tuple[int]]:
    if line_1[3] + deg_dif < line_2[3] or line_1[3] - deg_dif > line_2[3]: #Checks whether lines are not too far apart
        return []
    slp_1 = slope(line_1[0], line_1[1])
    slp_2 = slope(line_2[0], line_2[1])

    #Make sure check lines run in right direction towards other line
    if slp_1 != 'vertical' and slp_1 > -1 and slp_1 < 0: 
        check_dist_1 = check_dist
    else:
        check_dist_1 = check_dist
    if slp_2 == 'vertical' or slp_2 >= 0:
        check_dist_2 = -check_dist
    else:
        check_dist_2 = -check_dist

    #Find perpendicular lines and formulas for intersection lines. 
    perp_lines_1 = perp_line(line_1, check_dist_1, main_array=main_array, draw=draw)
    end_list = []
    for lines in perp_lines_1:
        intersec_points_1 = calc_intersec(lines, line_2)
        points_2 = sort_points(line_2[0], line_2[1])
        if intersec_points_1[0] >= points_2[0][0] and intersec_points_1[0] <= points_2[1][0] and intersec_points_1 != (0,0):
            if intersec_points_1 == (0,0):
                print(intersec_points_1, lines)
            end_list.append((lines[0], intersec_points_1)) 
    perp_lines_2 = perp_line(line_2, check_dist_2, main_array=main_array, draw=draw)
    for lines in perp_lines_2:
        intersec_points_2 = calc_intersec(lines, line_1)
        points_1 = sort_points(line_1[0], line_1[1])
        if intersec_points_2[0] >= points_1[0][0] and intersec_points_2[0] <= points_1[1][0] and intersec_points_2 != (0,0):
            end_list.append((lines[0], intersec_points_2)) 
            if intersec_points_2 == (0,0):
                print(intersec_points_2, lines)
    return end_list #Each set of two lines has two intersect lines, one the longest possible and one the shortest possible. Averages out later on.

#Calculates distance between two lines
def distance(line_1: tuple, line_2: tuple) -> list[int]:
    slp = slope(line_1[0], line_1[1])
    if slp == 'vertical':
        distance = abs(line_1[0][0]-line_2[0][0])
        return [distance, distance]
    elif slp == 0:
        distance = abs(line_1[0][1]-line_2[0][1])
        return [distance]

#Draws given line in given array. Used for visualization in png files.
def draw_line(main_array, line: tuple, colour=[0, 0, 255]) -> None:
    slp = slope(line[0], line[1])
    if slp == 'vertical':
        list_points = [line[0][1], line[1][1]]
        list_points.sort()
        for i in range(list_points[0], list_points[1]):
            main_array[i][line[0][0]] = np.array(colour)
    elif slp == 0:
        list_points = [line[0][0], line[1][0]]
        list_points.sort()
        for i in range(list_points[0], list_points[1]):
            main_array[line[0][1]][i] = np.array(colour) 
    else:
        b = calc_line_formula_b(slp, line[0][0], line[0][1])
        if slp < 1 and slp > -1:
            list_points = [line[0][0], line[1][0]]
            list_points.sort()
            for i in range(list_points[0], list_points[1]):
                y = (i*slp) + b
                round_y = round(y)
                main_array[round_y][i] = np.array(colour)
        else:
            y_a, y_b = convert_to_y(slp, b)
            list_points = [line[0][1], line[1][1]]
            list_points.sort()
            for i in range(list_points[0], list_points[1]):
                x = (i*y_a) + y_b
                round_x = round(x)
                main_array[i][round_x] = np.array(colour)

#Calculates the perpendicular lines of a given line. 
def perp_line(line: tuple, check_distance: float = -10, draw:bool = False, main_array = '') -> list[tuple[tuple[int]]]: #Form of return is [((start_x, start_y), (end_x, end_y)),((start_x, start_y), (end_x, end_y))]
    end_lines = []
    slp = slope(line[0],line[1])
    for point in [line[0], line[1]]:
        if slp == "vertical":
            perp_slp = 0
            end_point_x = calc_further_point(0, point[0] + check_distance, 0, main_array)
            end_point_y = point[1]
        elif slp == 0:
            perp_slp = "vertical"
            end_point_x = point[0] 
            end_point_y = calc_further_point(0, point[1] + check_distance, 0, main_array, x_or_y='y')
        else:
            perp_slp = -(1/slp)
            b = calc_line_formula_b(perp_slp, point[0], point[1])
            if slp < 1 and slp > -1:
                y_a, y_b = convert_to_y(perp_slp, b)
                end_point_x = calc_further_point(y_a, y_b, point[1] + check_distance, main_array)
                end_point_y = calc_further_point(0, point[1] + check_distance, 0, main_array, x_or_y='y')
            else:
                end_point_x = calc_further_point(0, point[0] + check_distance, 0, main_array)
                end_point_y = calc_further_point(perp_slp, b, point[0] + check_distance, main_array, x_or_y='y')
        perp_line = (point, (end_point_x, end_point_y))
        end_lines.append(perp_line)
        if draw:
            draw_line(main_array, perp_line, [255, 0, 0])
    return end_lines

#Calculation for the average length using processed image with skeletonization
def avg_len_calc(
        file_name: str, 
        param_list: list,
        colour: list[int], 
        visualize: bool = True,
        skeletonized: bool = False,
        ):
    
    pixel_to_microm = param_list[0][1]
    min_length_skeleton = param_list[1][1]
    list_width = param_list[2][1]
    deg_dif = param_list[3][1]
    check_distance = param_list[4][1]
    
    
    #open file
    image_file = cv2.imread(file_name)
    
    #skeletonize main file
    if skeletonized:
        skeleton = image_file
    else:
        # Convert to grayscale for processing
        gray_image = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
        structuring_element = square(1)

        # Apply binary opening and skeletonization
        preprocessed = binary_opening(gray_image, structuring_element)
        skeleton = skeletonize(preprocessed > 0)  # Skeletonize expects a binary image

        # Convert skeleton back to color (3 channels) to match original color structure
        skeleton = (skeleton * 255).astype(np.uint8)  # Scale back to 0-255
        skeleton = cv2.merge([skeleton, skeleton, skeleton])  # Make it 3-channel

        # Optionally, save visualization
        # if visualize:
        #     cv2.imwrite("skeleton.png", skeleton)

    #create file for later drawing of various results
    skeleton_draw = skeleton.copy()

    # Finding linear representation of z-lines
    z_line_list = []
    i = 0
    for x_line in skeleton: #loop over y coords
        j = 0
        for coordinate in x_line: #loop over x coords
            if list(skeleton[i][j]) == colour: #check if part of skeleton
                neighbs = neighbours(skeleton, j, i, colour)
                if single_neighbour(neighbs): #if only one neighbour then edge point so find other edge point
                    edge_point_1 = (j, i)
                    edge_point_2 = furthest_point(skeleton, j, i, colour)
                    length = math.dist(edge_point_1, edge_point_2)
                    if length > min_length_skeleton: #filter too short skeletons
                        angle_of_line = ang(length, edge_point_1, edge_point_2)
                        z_line_list.append((edge_point_1, edge_point_2, length, angle_of_line)) #fill list of z-lines
            j += 1
        i += 1

    # Matching of lines and transform to coords for dist calc
    match_list = []
    distance_list = []
    i = 0
    while i < len(z_line_list): #loop over the created list
        if visualize:
            draw_line(skeleton_draw, z_line_list[i]) #for visualization
        #created paramaters in order to not go out of bound with checking
        if i - list_width > 0: 
            start_point = i - list_width
        else:
            start_point = 0
        if i + list_width < len(z_line_list):
            end_point = i + list_width
        else:
            end_point = len(z_line_list)
        #calculate all z-line couples that adhere to the input parameters
        for j in range(start_point, end_point):
            # if z_line_list[i][0] == (0,0) or z_line_list[j][0] == (0,0):
            #     print(z_line_list[i], z_line_list[j])
            if abs(z_line_list[i][0][0] - z_line_list[j][0][0]) < check_distance and abs(z_line_list[i][0][1] - z_line_list[j][0][1]) < check_distance: # and abs(z_line_list[i][1][0] - z_line_list[j][0][0]) < check_distance and abs(z_line_list[i][1][1] - z_line_list[j][0][1]) < check_distance
                sorted_points = sort_points(z_line_list[i], z_line_list[j])
                match_list += check_if_cross(sorted_points[0], sorted_points[1], check_distance, deg_dif=deg_dif, main_array=skeleton_draw)
        i += 1

    #final distance calc
    for tuples in match_list:
        if visualize:
            rounded = (tuples[0], (round(tuples[1][0]), round(tuples[1][1])))
            draw_line(skeleton_draw, rounded, colour=[255,255,0]) #for visualization
        sarcomere_length = math.dist(tuples[0], tuples[1])
        if sarcomere_length > 1:
            distance_list.append(sarcomere_length)


    #result printing
    if len(distance_list) != 0:
        avg_distance = (sum(distance_list)/len(distance_list))/pixel_to_microm
        ratio = (len(distance_list)/2)/len(z_line_list)
    else:
        avg_distance = None
        ratio = len(z_line_list)

    if visualize:
        name = file_name.split(" ")[3]
        new = name.split(".")[0]
        new_file_name = "Results/Sacromere_mask/"+"line_skel_" + new + ".png"
        # cv2.imwrite(str(new_file_name), skeleton_draw)
    
    return (avg_distance, ratio, len(z_line_list), len(distance_list), skeleton_draw)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------FOURIER SECTION----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

#calculate bounds for the search space
def calculate_bounds(height: int, width: int, s_length: float, pixel_ratio: float) -> float:
    i = 1 
    while True:
        e = 2**i
        if e >= max(height,width):
            return e/(pixel_ratio* s_length)
        i += 1

def calc_circle_point(x, center_x, center_y, radius):  #unused
    y_k2 = radius**2 - (x-center_x)
    y = math.sqrt(y_k2) + center_y
    y_neg = -y
    return [(x,y), (x,y_neg)]

#calculate from polar to conventional axis coordinates
def polar_to_rec(radius: int, theta: float) -> tuple[int, int]:
    x = radius * math.cos(theta)
    y = radius * math.sin(theta)
    return (round(x), round(y))

#calc final distance from fourier to regular
def calc_microm_dist(height: int, width: int, four_dist: float, pixel_ratio: float) -> float:
    i = 1
    while True:
        e = 2**i
        if e >= max(height, width):
            return e/(pixel_ratio*four_dist)
        i += 1

#calc interval between drawn lines in the search circle
def set_interval(full: bool, interval_for_full: float, line_nr: int) -> float: 
    if full:
        return interval_for_full
    else:
        interval = (2 * math.pi)/line_nr
        return interval 

#calc average length of sarcomeres using a fourier transformed image
def four_calc(
        image_file: str, 
        param_list: list,
        visualize: bool = False
        ):
    
    pixel_to_microm = param_list[0][1]
    upper_s_len = param_list[1][1]
    lower_s_len = param_list[2][1]
    full_coverage = param_list[3][1]
    lines_from_center = param_list[4][1]
    interval_for_full_coverage = param_list[5][1] * math.pi
    
    transformed_image = cv2.imread(image_file)  
    draw_image = transformed_image.copy()

    #calculate search space boundaries and location
    lower_bounds = calculate_bounds(len(transformed_image), len(transformed_image[0]), upper_s_len, pixel_to_microm)
    upper_bounds = calculate_bounds(len(transformed_image), len(transformed_image[0]), lower_s_len, pixel_to_microm)
    center = (round(len(transformed_image[0])/2), round(len(transformed_image)/2))

    #start search
    i = lower_bounds
    brightness_dict = {}
    bright_list = []
    distance_list = []
    interval = set_interval(full_coverage, interval_for_full_coverage, lines_from_center)
    while i < upper_bounds: #radius interval
        j = 0
        total_brightness = 0.0
        k = 0
        while j < 2 * math.pi: #angle interval
            coords = polar_to_rec(i, j)
            center_x = coords[0] + center[0]
            center_y = coords[1] + center[1]
            place = transformed_image[center_y][center_x]
            if place[0] == place[2] and place[0] > 0: #check if place has been checked yet
                distance = math.dist(center, [center_x, center_y])
                distance = round(distance)
                try: #assign values to dict if it exists
                    brightness_dict[distance][0] = brightness_dict[distance][0] + int(place[0])
                    brightness_dict[distance][1] += 1 
                except KeyError: #create new dict if needed
                    brightness_dict[distance] = [int(place[0]), 1]
                total_brightness += place[0]
                k += 1
                transformed_image[center_y][center_x] = np.array([0, 0 , 255]) #change pixel to colour to show it has been checked
            j += interval
        total_brightness = total_brightness/k
        bright_list.append(total_brightness)
        distance_list.append(i)
        i += 1

    #create image with all checked pixel red as control
    # cv2.imwrite(image_file[:-4] + "_test.png", transformed_image)

    #create weight lists for every distance based on average brightness of the pixels
    weight_list = []
    dist_list = []
    brightness_list = []
    for distances in brightness_dict:
        weight = brightness_dict[distances][0]/brightness_dict[distances][1]
        weight_list.append((distances, weight))
        dist_list.append(distances)
        brightness_list.append(weight)

    # trim the list to only contain values around the maximum to be able to create an accurate distribution
    

    maximal = max(brightness_list)
    max_index = brightness_list.index(maximal)
    if max_index > 5:
        brightness_list = brightness_list[:min([len(brightness_list), (max_index *2)])]
        # brightness_list = brightness_list[max([0, (max_index - 20)]):]
        weight_list = weight_list[:min([len(brightness_list), (max_index *2)])]
        # weight_list = weight_list[max([0, (max_index - 20)]):]
    

    #create a final list which can be used to graph and fit a distribution
    final_list = []
    minimal = min(brightness_list)

    for elements in weight_list:
        loop_times = elements[1] - minimal
        i = 0
        while i < loop_times:
            final_list.append(elements[0])
            i += 1

    
    mu, std = stats.norm.fit(final_list)
    
    #calculate final value
    four_len = mu
    microm_len = calc_microm_dist(len(transformed_image), len(transformed_image[0]), four_len, pixel_to_microm)


    return microm_len

