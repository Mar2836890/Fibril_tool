# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:57:50 2021

@author: Linde Schoenmaker

Last edited on Jan 10 2025

@editor: Marjolein van Tol
"""
import cv2
from skimage.draw import line
from skimage import measure
import matplotlib.pyplot as plt
import scipy
import numpy as np
import math
from statistics import stdev

from skimage.morphology import skeletonize, binary_opening, binary_dilation, square
from statistics import median

#--------------Parameters-----------------------

global Ratio
global radius           # radius for crowded function
global max_distance     # original limit was set to 14, max distance between ends to get connected points
global limit1           # original limit was set to 16, to get the labels of all the centroids that are close to center
global limit2           # to get the labels of all the endpoints close to the endpoints


#--------------Code-----------------------
values = []        # for displaying plot with orientation distribution

def orientation(param0, param1):
#    if type(param1)!='int':
#        param1=param1[0]
#    if type(param0)!='int':
#        param0=param0[0]
    if abs(param1) > 0:
        orient = math.atan(-param0 / param1)
    else:
        orient = math.pi / 2

    return orient

def touching_border(objects):
    seed = np.ones(objects.shape, dtype=int)
    sarc_length = 12
    seed[sarc_length:-sarc_length, sarc_length:-sarc_length] = 0
    touching = scipy.ndimage.binary_propagation(
        seed, mask=objects, structure=np.ones((3, 3))
    ).astype(int)
    removed = np.logical_xor(objects, touching)
    no_border = np.zeros(objects.shape, dtype=int)
    no_border[sarc_length:-sarc_length, sarc_length:-sarc_length] = removed[
        sarc_length:-sarc_length, sarc_length:-sarc_length
    ]

    return no_border


def calc_length(binary):
    contours, __ = cv2.findContours(binary.astype(np.uint8), 1, 2)
    lengths = []

    for cnt in contours:
        length = cv2.arcLength(cnt, True) / 2
        lengths.append(length)

    std = None
    average = None
    if len(lengths) > 0:
        average = sum(lengths) / len(lengths)
    if len(lengths) > 1:
        std= stdev(lengths)

    return average, std


def calc_ratio(centroids, connect_fibrils):
    # does not take into account ones touching the borders
    total = np.count_nonzero(centroids)
    # which are present in both
    positive = centroids * connect_fibrils
    numpositive = np.count_nonzero(positive)
    perc = None
    if total>0:
        perc = numpositive / total * 100

    return perc


def calc_width(preprocessed, labels_image, connect_fibrils, connected_lab):
    # width function
    fib_inv = 1 - connect_fibrils
    dist = cv2.distanceTransform(fib_inv.astype(np.uint8), cv2.DIST_L2, 3)
    # disk_dist = dist * preprocessed

    propsa = measure.regionprops(labels_image, dist)
    max_dist = []

    for j, label in enumerate(propsa):
        # only keep z-disks that touch a fibril
        if label.min_intensity == 0:
            # save label & max intensity
            max_dist.append(label.max_intensity)

    avg_width = None    
    if len(max_dist) > 0:
        avg_width = sum(max_dist) / len(max_dist)
    
    std_width = None
    if len(max_dist) > 1:
        arr = np.array(max_dist)
        std_width = arr.std()
    return avg_width, std_width


def orient_distr(lines, centroids, file, visualize=False):
    fibr_centroids = lines * centroids
    unique_fib, counts_fib = np.unique(fibr_centroids, return_counts=True)
    num_centroids = dict(zip(unique_fib, counts_fib))
    del num_centroids[0]
    if not num_centroids.values():
        p_val = None
        return p_val
        
    max_num = max(num_centroids.values())
    if max_num == 0:
        p_val = None
        return p_val
    norm_lengths = {k: v / max_num for k, v in num_centroids.items()}

    if visualize:
        orient_lines = np.zeros(lines.shape, np.uint8)

    orient_values = {}
    propsa = measure.regionprops(lines)
    lm = measure.LineModelND()

    for j, label in enumerate(propsa):
        lm.estimate(label.coords)
        orient_val = orientation(lm.params[1][0], lm.params[1][1])

        if orient_val <= -3 * math.pi / 8 or orient_val >= 3 * math.pi / 8:
            binned_value = 1
        elif math.pi / 8 <= orient_val < 3 * math.pi / 8:
            binned_value = 2
        elif -math.pi / 8 <= orient_val < math.pi / 8:
            binned_value = 3
        elif -3 * math.pi / 8 <= orient_val < -math.pi / 8:
            binned_value = 4
        orient_values[label.label] = binned_value

        if visualize:
            try:
                r0 = label.coords[0, 0]
                c0 = lm.predict_y([r0])
                r1 = label.coords[-1, 0]
                c1 = lm.predict_y([r1])
                rr, cc = line(r0, int(c0), r1, int(c1))
                orient_lines[rr, cc] = (
                    orientation(lm.params[1][0],
                                lm.params[1][1]) * 180 / math.pi
                )
            except ValueError:
                continue
            except IndexError:
                continue

    # calculate weighted frequencies
    data = {1: 0, 2: 0, 3: 0, 4: 0}
    weighted_data = {1: 0, 2: 0, 3: 0, 4: 0}
    for k in norm_lengths:
        data[orient_values[k]] += 1
        weighted_data[orient_values[k]] += norm_lengths[k]

    counts = list(data.values())
    weighted_counts = list(weighted_data.values())

    if all(i >= 5 for i in counts):
        __, p_val = scipy.stats.chisquare(list(weighted_counts))
    else:
        p_val = None

    # display results
    names = list(data.keys())
    values.append(weighted_counts)
    # plt.bar(names, weighted_counts)

    # plt.grid(axis="y", alpha=0.75)
    # plt.xlabel("Orientation")
    # plt.title(file)
    # plt.ylabel("Frequency")

    # my_xticks = ["|", "/", "-", "\\"]
    # plt.xticks(names, my_xticks)
    
    # plt.savefig(file+"_orientation.png", dpi=100)
    # plt.close()

    return p_val




def angle(a, b, c):
    try:
        ang = math.degrees(math.acos((c ** 2 - b ** 2 - a ** 2) / (-2.0 * a * b)))
    except ValueError:
        ang = 180

        if abs(c - a - b) > 1e-10:
            print("Warning: possible problem calculating the angle")
            print(f"c - a - b = {c - a - b}")

    return ang


def get_preprocessed(mask, propagate=False):
    # perform opening
    
    structuring_element = square(1)
    
    
    preprocessed = binary_opening(mask, structuring_element)

    # propagate within original mask to have original zdisk shapes
    if propagate:
        preprocessed = scipy.ndimage.morphology.binary_propagation(
            preprocessed, mask=mask
        )

    return preprocessed.astype(np.uint8)


def get_cclabel(img):
    img = img.astype(np.uint8)
    __, labels_im = cv2.connectedComponents(img)

    return labels_im.astype(np.uint16)


def get_sparse(img_labeled):
    cols = np.arange(img_labeled.size)
    sparse_labels = scipy.sparse.csr_matrix(
        (cols, (img_labeled.ravel(), cols)),
        shape=(img_labeled.max() + 1, img_labeled.size),
    )
    sparse_coords = [
        np.unravel_index(row.data, img_labeled.shape) for row in sparse_labels
    ]

    return sparse_coords


def get_orientation(img_labeled, skel_crit=True):
    props = measure.regionprops(img_labeled)
    lm = measure.LineModelND()
    orient = np.zeros(np.max(img_labeled) + 1)

    for j, obj in enumerate(props):
        if skel_crit and obj.area <= 3:
            orient[obj.label] = 2
        else:
            lm.estimate(obj.coords)
            orient[obj.label] = orientation(lm.params[1][0], lm.params[1][1])

    return orient


def get_centroids(mask):
    centroids = np.zeros(mask.shape, dtype=np.uint8)

    # find contours in the binary image
    contours, __ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for j, c in enumerate(contours):
        # calculate moments for each contour using their coordinates
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            continue
        centroids[cY, cX] = 1

    return centroids


def detect_crowded(image, diameter):
    
    # use circular kernal to count number of object pixels around center
    # if this number is high then pixel is in crowded location
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    kernel = kernel  # / np.count_nonzero(kernel)
    convolved = cv2.filter2D(
        image.astype(np.float32),
        -1,
        kernel.astype(np.float32),
        borderType=cv2.BORDER_CONSTANT,
    )
    # if 4 objects in circle then crowded
    ret, thresh = cv2.threshold(convolved, 4, 255, cv2.THRESH_BINARY)
    regions_dilated = binary_dilation(thresh, kernel)

    return regions_dilated


def get_endpoints(fibr_bin):
    se1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    se2 = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]])
    endpoints1 = scipy.ndimage.binary_hit_or_miss(fibr_bin, se1, se2)
    endpoints2 = scipy.ndimage.binary_hit_or_miss(fibr_bin, se1, np.rot90(se2))
    endpoints3 = scipy.ndimage.binary_hit_or_miss(fibr_bin, se1, np.rot90(se2, 2))
    endpoints4 = scipy.ndimage.binary_hit_or_miss(fibr_bin, se1, np.rot90(se2, 3))

    combined = endpoints1 | endpoints2 | endpoints3 | endpoints4

    ends = combined.astype(int)

    return ends


def get_neighb(
    center,
    labels,
    sparse_labels,
    sparse_skeleton,
    orient,
    connections,
    fibr_label,
    fibrs,
    lines,
    orient_grid,
    limit1,
    root=True,
):
    result = sparse_labels[center]
    
    # to get the labels of all the centroids that are close to center
    limit = limit1
    
    borderx=result[0]
    bordery=result[1]
    if type(borderx)!='int':
#        print("it is an array.result[0]=", result[0])
        borderx=result[0][0]

        
    if type(bordery)!='int':
        bordery=result[1][0]


    if (borderx - limit) > 0:
        xmin = borderx - limit
    else:
        xmin = 0
    if (borderx + limit) < labels.shape[0]:
        xmax = borderx + limit
    else:
        xmax = borderx
#        print("I'm at the border of xmax")
    if (bordery - limit) > 0:
        ymin = bordery - limit
    else:
        ymin = 0
    if (bordery + limit) < labels.shape[1]:
        ymax = bordery + limit
    else:
        ymax = bordery
#        print("I'm at the border of xmax")
#    print("xmax= ",xmax, " ymax= ", ymax, "labels.shape[0]= ", labels.shape[0], "labels.shape[1]", labels.shape[1])    
    n1, n2 = np.arange(xmin, xmax), np.arange(ymin, ymax)
    mask = labels[n1[:, None], n2[None, :]]

    # keep all the labels for which a label is not yet assigned
    unassigned = fibrs[n1[:, None], n2[None, :]]
    mask_unass = mask[(unassigned == 1)]
    uniques = np.unique(mask_unass)
    uniques = uniques[(uniques != 0)]
    neighbs = uniques[(uniques != center)]

    crit_met = []

    for neighb in neighbs:
        thet1 = orient[neighb]
        thet2 = orient[center]

        # do not use orientation if one of the disks is too round
        if thet1 == 2 or thet2 == 2:
            diff = 0
        else:
            diff = min(abs(thet1 - thet2), math.pi - abs(thet1 - thet2))

        if diff <= math.pi / 6:
            coords = sparse_skeleton[neighb]
            best_dist = np.inf

            for i in range(0, len(coords[0])):
                dist = math.sqrt(
                    (borderx - coords[0][i]) ** 2 + (bordery - coords[1][i]) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
            if 7 <= best_dist <= 13:
                crit_met.append(neighb)

    if root:
        return select_root(
            crit_met,
            fibrs,
            result,
            lines,
            connections,
            sparse_labels,
            fibr_label,
            center,
            orient_grid,
        )
    else:
        return continue_fibril(
            crit_met,
            fibrs,
            result,
            lines,
            connections,
            sparse_labels,
            fibr_label,
            center,
            orient_grid,
        )


def select_root(
    crit_met,
    fibrs,
    result,
    lines,
    connections,
    sparse_labels,
    fibr_label,
    center,
    orient_grid,
):
    if len(crit_met) == 0:
        fibrs[result[0], result[1]] = 2
    elif len(crit_met) == 1:
        # do not start fibrils at the end of the fibril, only at middle
        crit_met = []
    elif len(crit_met) == 2:
        conditions_met = False
        # get location of previous and neighbor
        prev = sparse_labels[crit_met[0]]
        nex = sparse_labels[crit_met[1]]

        borderx=result[0]
        bordery=result[1]
        if type(borderx)!='int':
            borderx=result[0][0]
        
        if type(bordery)!='int':
            bordery=result[1][0]
        
        prevx=prev[0]
        prevy=prev[1]
        if type(prevx)!='int':
            prevx=prev[0][0]
        
        if type(prevy)!='int':
            prevy=prev[1][0]
           
           
        nexx=nex[0]
        nexy=nex[1]
        if type(nexx)!='int':
            nexx=nex[0][0]
        
        if type(nexy)!='int':
            nexy=nex[1][0]
        # determine length of sides triangle
        opp = math.sqrt((prevx - nexx) ** 2 + (prevy - nexy) ** 2)
        apr = math.sqrt((prevx - borderx) ** 2 + (prevy - bordery) ** 2)
        ane = math.sqrt((nexx - borderx) ** 2 + (nexy - bordery) ** 2)

        # calc angle between previous, center and next
        ang = angle(ane, apr, opp)

        orient_region = orient_grid[center]

        # for crowded region determine if resulting fibril is perpendicular to zdisks
        if orient_region == 2 and ang > 135:
            conditions_met = True
        elif ang > 135:
            # calc orientation of resulting fibril
            orient_line = orientation(prevx - nexx, prevy - nexy)
            # compare to orientation in region
            diff = abs(orient_region - orient_line)  # minus orient fibril
            # use this as criterium
            if math.pi * 1 / 4 < diff < math.pi * 3 / 4:
                conditions_met = True

        # check if criteria are met
        if conditions_met:
            fibrs[result[0], result[1]] = fibr_label
            fibrs[prev[0], prev[1]] = fibr_label
            rr, cc = line(int(result[0]), int(result[1]), int(prev[0]), int(prev[1]))
            lines[rr, cc] = fibr_label
            connections[center, crit_met[0]] = True
            connections[crit_met[0], center] = True

            fibrs[nex[0], nex[1]] = fibr_label
            rr, cc = line(int(result[0]), int(result[1]), int(nex[0]), int(nex[1]))
            lines[rr, cc] = fibr_label
            connections[center, crit_met[1]] = True
            connections[crit_met[1], center] = True
        else:
            crit_met = []

    elif len(crit_met) > 2:
        best = np.inf
        conditions_met = False
        # All possible pairs in List
        # Using list comprehension + enumerate()
        res = [(a, b) for idx, a in enumerate(crit_met) for b in crit_met[idx + 1 :]]

        orient_region = orient_grid[center]

        if orient_region == 2:
            for i, comb in enumerate(res):
                prev = sparse_labels[comb[0]]
                nex = sparse_labels[comb[1]]
                # determine length of sides triangle
                opp = math.sqrt((prev[0] - nex[0]) ** 2 + (prev[1] - nex[1]) ** 2)
                apr = math.sqrt((prev[0] - result[0]) ** 2 + (prev[1] - result[1]) ** 2)
                ane = math.sqrt((nex[0] - result[0]) ** 2 + (nex[1] - result[1]) ** 2)
                # calc angle between previous, center and next
                ang = angle(ane, apr, opp)
                if 180 - ang < best:
                    best = 180 - ang
                    j = i
            if best <= 45:
                conditions_met = True
        else:
            for i, comb in enumerate(res):
                prev = sparse_labels[comb[0]]
                nex = sparse_labels[comb[1]]
                
                prevx=prev[0]
                prevy=prev[1]
                if type(prevx)!='int':
                    prevx=prev[0][0]
                
                if type(prevy)!='int':
                    prevy=prev[1][0]
                    
                    
                nexx=nex[0]
                nexy=nex[1]
                if type(nexx)!='int':
                    nexx=nex[0][0]
                
                if type(nexy)!='int':
                    nexy=nex[1][0]
                
                # determine length of sides triangle
                opp = math.sqrt((prevx - nexx) ** 2 + (prevy - nexy) ** 2)
                apr = math.sqrt((prevx - result[0]) ** 2 + (prevy - result[1]) ** 2)
                ane = math.sqrt((nexx - result[0]) ** 2 + (nexy - result[1]) ** 2)
                # calc angle between previous, center and next
                ang = angle(ane, apr, opp)
                if ang > 135:
                    # calc orientation of resulting fibril
                    orient_line = orientation(prevx - nexx, prevy - nexy)
                    # compare to orientation in region
                    # minus orient fibril
                    diff = abs(orient_region - orient_line)
                    # use this as criterium
                    if abs(diff - math.pi / 2) < best:
                        best = abs(diff - math.pi / 2)
                        j = i

            if math.pi * 1 / 4 < best + math.pi / 2 < math.pi * 3 / 4:
                conditions_met = True

        if conditions_met:
            fibrs[result[0], result[1]] = fibr_label
            for neighb in res[j]:
                coords = sparse_labels[neighb]
                fibrs[coords[0], coords[1]] = fibr_label
                rr, cc = line(
                    int(result[0]), int(result[1]), int(coords[0]), int(coords[1])
                )
                lines[rr, cc] = fibr_label
                connections[center, neighb] = True
                connections[neighb, center] = True
            crit_met = [*res[j]]
        else:
            crit_met = []

    return crit_met, fibrs, lines, connections


def continue_fibril(
    crit_met,
    fibrs,
    result,
    lines,
    connections,
    sparse_labels,
    fibr_label,
    center,
    orient_grid,
):
    if len(crit_met) == 1:
        conditions_met = False
        # if only one neighbor, assign fibril label
        neigh = crit_met[0]
        # get location of previous and neighbor
        prev_label = np.where(connections[center, :].T)
        prev = sparse_labels[prev_label[0][0]]
        nex = sparse_labels[neigh]
        
        prevx=prev[0]
        prevy=prev[1]
#        if type(prevx)!='int':
#            prevx=prev[0][0]
#        
#        if type(prevy)!='int':
#            prevy=prev[1][0]
            
            
        nexx=nex[0]
        nexy=nex[1]
#        if type(nexx)!='int':
#            nexx=nex[0][0]
#        
#        if type(nexy)!='int':
#            nexy=nex[1][0]
        
        # determine length of sides triangle
        opp = math.sqrt((prevx - nexx) ** 2 + (prevy - nexy) ** 2)
        apr = math.sqrt((prevx - result[0]) ** 2 + (prevy - result[1]) ** 2)
        ane = math.sqrt((nexx - result[0]) ** 2 + (nexy - result[1]) ** 2)
        # calc angle between previous, center and next
        ang = angle(ane, apr, opp)

        orient_region = orient_grid[center]

        if orient_region == 2 and ang > 135:
            conditions_met = True
        elif ang > 135:
            # calc orientation of resulting fibril
            orient_line = orientation(prevx - nexx, prevy - nexy)
            # compare to orientation in region
            diff = abs(orient_region - orient_line)  # minus orient fibril
            # use this as criterium
            if math.pi * 1 / 4 < diff < math.pi * 3 / 4:
                conditions_met = True

        if conditions_met:
            coords = nex
            fibrs[nexx, nexy] = fibr_label
            rr, cc = line(
                int(result[0]), int(result[1]), int(nexx), int(nexy)
            )
            lines[rr, cc] = fibr_label
            connections[center, neigh] = True
            connections[neigh, center] = True
        else:
            crit_met = []

    elif len(crit_met) > 1:
        # only select one neighbor
        # determine which
        best = np.inf
        conditions_met = False

        orient_region = orient_grid[center]

        if orient_region == 2:
            for i, neigh in enumerate(crit_met):
                # get location of previous and neighbor
                prev_label = np.where(connections[center, :].T)
                prev = sparse_labels[prev_label[0][0]]
                nex = sparse_labels[neigh]
                # determine length of sides triangle
                opp = math.sqrt((prev[0] - nex[0]) ** 2 + (prev[1] - nex[1]) ** 2)
                apr = math.sqrt((prev[0] - result[0]) ** 2 + (prev[1] - result[1]) ** 2)
                ane = math.sqrt((nex[0] - result[0]) ** 2 + (nex[1] - result[1]) ** 2)
                # calc angle between previous, center and next
                ang = angle(ane, apr, opp)
                if 180 - ang < best:
                    best = 180 - ang
                    j = i
            if best <= 45:
                conditions_met = True
        else:
            for i, neigh in enumerate(crit_met):
                # get location of previous and neighbor
                prev_label = np.where(connections[center, :].T)
                prev = sparse_labels[prev_label[0][0]]
                nex = sparse_labels[neigh]
                # determine length of sides triangle
                opp = math.sqrt((prev[0] - nex[0]) ** 2 + (prev[1] - nex[1]) ** 2)
                apr = math.sqrt((prev[0] - result[0]) ** 2 + (prev[1] - result[1]) ** 2)
                ane = math.sqrt((nex[0] - result[0]) ** 2 + (nex[1] - result[1]) ** 2)
                # calc angle between previous, center and next
                ang = angle(ane, apr, opp)
                if ang > 135:
                    # calc orientation of resulting fibril
                    orient_line = orientation(prev[0] - nex[0], prev[1] - nex[1])
                    # compare to orientation in region
                    # minus orient fibril
                    diff = abs(orient_region - orient_line)
                    # use this as criterium
                    if abs(diff - math.pi / 2) < best:
                        best = abs(diff - math.pi / 2)
                        j = i

            if math.pi * 1 / 4 < best + math.pi / 2 < math.pi * 3 / 4:
                conditions_met = True

        if conditions_met:
            neighb = crit_met[j]
            coords = sparse_labels[neighb]
            fibrs[coords[0], coords[1]] = fibr_label
            rr, cc = line(
                int(result[0]), int(result[1]), int(coords[0]), int(coords[1])
            )
            lines[rr, cc] = fibr_label
            connections[center, neighb] = True
            connections[neighb, center] = True
            crit_met = [neighb]
        else:
            crit_met = []

    return crit_met, fibrs, lines, connections


def trace_fibr(cent, labels, sparse_labels, sparse_skeleton, orient, limit1, orient_grid=None):

    lines = np.zeros(cent.shape, np.uint16)
    connections = np.zeros((np.max(labels) + 1, np.max(labels) + 1), bool)

    # for checking if centroid is already assigned
    fibrs = cent.copy().astype(np.uint16)

    # list with unassigned centroids to iterate over
    unass = np.unique(labels)
    unass = list(unass[(unass != 0)])

    fibr_label = 1
    while len(unass) > 0:
        # if fibr_label > 13:
        #    break
        # root_lab is the label number
        root_lab = unass.pop(0)
        neighbs, fibrs, lines, connections = get_neighb(
            root_lab,
            labels,
            sparse_labels,
            sparse_skeleton,
            orient,
            connections,
            fibr_label + 1,
            fibrs,
            lines,
            orient_grid,
            limit1,
            True,
        )
        if len(neighbs) > 0:
            fibr_label += 1
            # iterate over neighbors
            while len(neighbs) > 0:
                new_root = neighbs.pop(0)
                # try to remove new root from list,
                # will not work when centerpoint already visited
                try:
                    unass.remove(new_root)
                except ValueError:
                    continue
                # find neighbors of the new root
                new_neighbs, fibrs, lines, connected = get_neighb(
                    new_root,
                    labels,
                    sparse_labels,
                    sparse_skeleton,
                    orient,
                    connections,
                    fibr_label,
                    fibrs,
                    lines,
                    orient_grid,
                    False,
                )
                neighbs.extend(new_neighbs)
                # remove duplicates
                neighbs = list(dict.fromkeys(neighbs))

    return lines, connections


def get_connected(
    ends, ends_labeled, ends_sparse, fibrils_binary, centroids_sparse, connections, max_distance, limit2
):
    # for each fibril
    assigned = np.zeros(fibrils_binary.shape, dtype=np.uint8)
    unass = np.unique(ends_labeled)
    unass = list(unass[(unass != 0)])
    while len(unass) > 0:
        # get coordinates endpoint(s)
        center = unass.pop(0)
        result = ends_sparse[center]
        # to get the labels of all the endpoints close to the endpoints
        limit = limit2

        if result[0] - limit > 0:
            xmin = result[0] - limit
        else:
            xmin = 0
        if result[0] + limit < ends.shape[0]:
            xmax = result[0] + limit
        else:
            xmax = ends.shape[0]
        if result[1] - limit > 0:
            ymin = result[1] - limit
        else:
            ymin = 0
        if result[1] + limit < ends.shape[1]:
            ymax = result[1] + limit
        else:
            ymax = ends.shape[1]
        n1, n2 = np.arange(xmin, xmax), np.arange(ymin, ymax)
        mask = ends_labeled[n1[:, None], n2[None, :]]
        mask_assigned = assigned[n1[:, None], n2[None, :]]
        mask_unass = mask[(mask_assigned == 0)]
        uniques = np.unique(mask_unass)
        uniques = uniques[(uniques != 0)]
        neighbs = uniques[(uniques != center)]

        if len(neighbs) > 0:
            prev_lab = np.where(connections[center, :].T)
            prev = centroids_sparse[prev_lab[0][0]]

        best = np.inf
        for i, neighb in enumerate(neighbs):
            coords = ends_sparse[neighb]
            dist = math.sqrt(
                (result[0] - coords[0]) ** 2 + (result[1] - coords[1]) ** 2
            )
            if dist <= max_distance:
                # angle with proint connected to the root endpoint
                opp = math.sqrt((prev[0] - coords[0]) ** 2 + (prev[1] - coords[1]) ** 2)
                apr = math.sqrt((prev[0] - result[0]) ** 2 + (prev[1] - result[1]) ** 2)
                ane = math.sqrt(
                    (coords[0] - result[0]) ** 2 + (coords[1] - result[1]) ** 2
                )

                ang1 = angle(ane, apr, opp)

                # angle with proint connected to the neighbor endpoint
                nex_lab = np.where(connections[neighb, :].T)
                nex = centroids_sparse[nex_lab[0][0]]

                opp = math.sqrt((nex[0] - result[0]) ** 2 + (nex[1] - result[1]) ** 2)
                apr = math.sqrt((nex[0] - coords[0]) ** 2 + (nex[1] - coords[1]) ** 2)

                ang2 = angle(ane, apr, opp)

                if ang1 > 120 and ang2 > 120:
                    if abs(ang1 - ang2) < 40:
                        if 360 - ang1 - ang2 < best:
                            best = 360 - ang1 - ang2
                            j = i
        if best < 360:
            coords = ends_sparse[neighbs[j]]
            rr, cc = line(
                int(result[0]), int(result[1]), int(coords[0]), int(coords[1])
            )
            fibrils_binary[rr, cc] = 1
            assigned[rr, cc] = 1
            try:
                unass.remove(neighbs[j])
            except ValueError:
                continue

    return fibrils_binary


def orientation_circle(
    centroids_crow, centroids_sparse, centroids_labeled, zdisk_orient
):
    grid_orient = np.zeros(np.max(centroids_labeled) + 1)
    # set default orientation to 2
    grid_orient.fill(2)

    orientation_img = np.zeros(centroids_labeled.shape, dtype=np.float32)

    centroids = np.unique(centroids_crow)
    centroids = centroids[(centroids != 0)]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

    side = np.zeros((15, centroids_labeled.shape[1]))
    centroids_labeled = np.concatenate((centroids_labeled, side), axis=0)
    centroids_labeled = np.concatenate((side, centroids_labeled), axis=0)
    side = np.zeros((centroids_labeled.shape[0], 15))
    centroids_labeled = np.concatenate((centroids_labeled, side), axis=1)
    centroids_labeled = np.concatenate((side, centroids_labeled), axis=1)

    for centroid in centroids:
        orientations = []
        coords = centroids_sparse[centroid]
        coordx=coords[0]
        coordy=coords[1]
        if type(coordx)!='int':
            coordx=coords[0][0]
        
        if type(coordy)!='int':
            coordy=coords[1][0]
        # make mask with 1's in cells in circle around the centroid coordinates
        centroids_mask = centroids_labeled[
            int(coordx) : int(coordx + 31), int(coordy) : int(coordy + 31)
        ]

        centroids_mask = centroids_mask * kernel
        values = np.unique(centroids_mask)
        values = values[(values != 0)]
        for value in values:
            orient = zdisk_orient[int(value)]
            if orient != 2:
                orientations.append(orient)

        if len(orientations) == 0:
            continue

        med_orient = median(orientations)
        grid_orient[centroid] = med_orient
        orientation_img[coords] = (med_orient + 2) * 60

    return grid_orient, orientation_img


def imshow_components(original, lines):
    # make lines thicker
    element = np.ones((2, 2), np.uint8)
    lines = cv2.dilate(lines, element)

    # for lines
    lines_hue = np.uint8((lines - np.floor(lines / 20) * 20) * 12 + 1)
    # make background black
    lines_hue[lines == 0] = 0
    blank_ch = 255 * np.ones_like(lines_hue)
    labeled_img = cv2.merge([lines_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # overlay lines & original
    labeled_img[lines_hue == 0] = original[lines_hue == 0]

    #prev = cv2.imread("F:/RP2/results/FM_Start 19-11-2019_Plate_D_p00_0_B02f02d2.tif")
    # prev = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)

#    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
#
#    # axes.imshow(labeled_img)
#    # axes.set_title('Myofibrils', fontsize=20)
#
#    ax = axes.ravel()
#
#    ax[0].imshow(original)
#    ax[0].axis("off")
#    ax[0].set_title("original", fontsize=20)
#
#    ax[1].imshow(labeled_img)
#    ax[1].axis("off")
#    ax[1].set_title("grid", fontsize=20)
#
#    fig.tight_layout()
#    # fig.show()

    return original, labeled_img

def detect_fibril(folder_name, file, param_list):
    #--------------Parameters-----------------------

    Ratio = param_list[0][1]              
    # Scale = Ratio/5.6           

    radius = int(param_list[1][1])  
    diameter = (radius*2) +1   
    max_distance = int((param_list[2][1]))   

    limit1 = int(param_list[3][1])    
    limit2 = int(param_list[4][1])  

    #--------------Output Values-----------------------
    num_fibrils=0
    average=0
    std=0
    avg_width=0
    std_width=0
    p_val=0
    pos_centroids=0
    
    
    print("process file: ", file)
    
    img_dir = folder_name+'/'+file
    
    image = cv2.imread(img_dir) 

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # remove small z-disks
    preprocessed = get_preprocessed(image, propagate=False)

    preprocessed_labeled = get_cclabel(preprocessed)
    
    # skeletonize
    skeleton = skeletonize(preprocessed)
    skeleton_labeled = preprocessed_labeled * skeleton
    skeleton_sparse = get_sparse(skeleton_labeled)
    
    # fitline and orientation
    orientation = get_orientation(skeleton_labeled)

    # get centroids
    centroids = get_centroids(preprocessed)
    centroids_labeled = preprocessed_labeled * centroids
    centroids_sparse = get_sparse(centroids_labeled)

    # detect crowded regions
    regions_crowded = detect_crowded(centroids, diameter)
    centroids_crow = centroids_labeled * regions_crowded

    # get the grid orientations
    # regions_labeled = get_cclabel(regions_crowded)
    grid_orient, orientation_img = orientation_circle(
        centroids_crow, centroids_sparse, centroids_labeled, orientation
    )

    # trace myofibrils in normal & crowded regions
    fibrils, connections = trace_fibr(
        centroids,
        centroids_labeled,
        centroids_sparse,
        skeleton_sparse,
        orientation,
        limit1,
        grid_orient,
    )

    # get info for connecting endpoints
    fibrils_binary = cv2.threshold(fibrils, 1, 1, cv2.THRESH_BINARY)[1]
    ends = get_endpoints(fibrils)
    # when have something for crowded use both here
    ends_labeled = ends * centroids_labeled
    ends_sparse = get_sparse(ends_labeled)

    # connect endpoints & relabel
    connected_fibrils = get_connected(
        ends,
        ends_labeled,
        ends_sparse,
        fibrils_binary,
        centroids_sparse,
        connections,
        max_distance,
        limit2
    )
    connected_labeled = get_cclabel(connected_fibrils)

    # calculate features if there are fibrils
    if np.max(connected_labeled) > 0:

        # calc average width of fibrils
        avg_width, std_width = calc_width(
            preprocessed, preprocessed_labeled, connected_fibrils, connected_labeled
        )

        # calc ratio centroids to connected centroids
        pos_centroids = calc_ratio(centroids, connected_fibrils)

        # remove fibrils touching the border
        primaryobjects = touching_border(connected_fibrils)
        primary_labeled = get_cclabel(primaryobjects)
        num_fibrils = np.max(primary_labeled)
#        print(f"Number of fibrils: {num_fibrils}")

        # calc length features and orientation
        average, std = calc_length(primaryobjects)

        p_val = orient_distr(primary_labeled, centroids, img_dir[:-4])

    img_preprocessed, img_fibrils = imshow_components(
        preprocessed * 255, connected_labeled
    )
    # cv2.imwrite(folder_name + '/Fibril_mask/' + f"test_{file[:-4]}.tif", preprocessed * 255)

    # cv2.imwrite("Results/Fibril_mask/" + file[:-4] + "_fibrils.png", img_fibrils.astype(np.uint8))
    return file, num_fibrils, average/Ratio, std/Ratio, avg_width/Ratio, std_width/Ratio, p_val, pos_centroids, values[0][0], values[0][1], values[0][2], values[0][3], img_fibrils.astype(np.uint8)
