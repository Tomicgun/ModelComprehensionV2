################################################################################
# Authors: Thomas Marten, Prof. Bastian Tenbergen, Kritika Parajuli            #
# This code was written for a SUNY Oswego Cyper Physical research lab          #
#                                                                              #
# This code also uses the code of a stack overflow user by the name of cosmic  #
# and his post at the link bellow:                                             #
# https://stackoverflow.com/questions/51429596/centroidal-voronoi-tessellation #
#                                                                              #
################################################################################


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import Voronoi
import sys
import cv2 as cv
import pdf2image as p2i
import pandas as pd
import os
import easyocr
import matplotlib.pyplot as plt


def pdf_2_image(pdf_file_location,pdf_name):
    pages = p2i.convert_from_path(pdf_file_location +'/'+pdf_name+'.pdf')
    for page in pages:
        page.save('png files\\' + pdf_name + '.png','PNG')
    return 'png files\\' + pdf_name + '.png'

def distance(points):
    distances = []
    for point1 in points:
        for point2 in points:
            if point1[0] != point2[0] and point1[1] != point2[1]:
                distances.append(np.linalg.norm(point1-point2))
    return distances

def find_contour(pdf_file_location,pdf_name,blur_int,K):
    points = []

    # getting the image from pdf
    image = cv.imread(pdf_2_image(pdf_file_location,pdf_name))

    # convert to gray scale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    #create blur
    blur = cv.GaussianBlur(gray, (blur_int, blur_int),cv.BORDER_DEFAULT)

    # apply blur and filter out pixels
    ret, thresh = cv.threshold(blur, 200, 255,cv.THRESH_BINARY_INV)

    #writing the image to local disk
    #cv.imwrite("thresh {name}.png".format(name = pdf_name),thresh)

    contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    blank = np.zeros(thresh.shape[:2],dtype='uint8')

    cv.drawContours(blank, contours, -1,(255, 0, 0), 1)

    #cv.imwrite('contours/' + "Contours {name}.png".format(name = pdf_name), blank)

    for i in contours:
        M = cv.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv.drawContours(image, [i], -1, (0, 255, 0), 2)
            cv.circle(image, (cx, cy), 7, (0, 0, 255), -1)
            cv.putText(image, "center", (cx - 20, cy - 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            points.append((cx,cy))
            #print(f"x: {cx} y: {cy}")

    cv.imwrite('contours/' + "Contours {name}.png".format(name = pdf_name), image)
    return cluster_points(points,K,pdf_name=pdf_name)

def find_tessellation(pdf_file_location, pdf_name, blur_int):
    df = pd.read_csv('Name to Nodes.csv')
    df.columns = ['Name','Nodes']
    df = df[df['Name'] == pdf_name]
    if df.shape[0] == 0:
        print('PDF not in table')
        return 0,0,0
    K = df.values.flatten().tolist()[1]
    tesselation = find_contour(pdf_file_location,pdf_name,blur_int,K)
    if len(tesselation) == 0:
        print('No center points found')
        return 0,0,0
    X, Y = np.hsplit(tesselation,2)
    bounding_box = [0.,int(np.max(X)+200), 0., int(np.max(Y)+200)]
    # do the tesselation thingy
    centroids = plot(tesselation,bounding_box)
    if len(centroids) == 1:
        dis = 0
        stan_dev = 0
        average = 0
        print('Voronoi Tesselation with only one center point')
    elif len(centroids) == 0:
        dis = 0
        stan_dev = 0
        average = 0
        print('Voronoi not calculated')
    else:
        dis = distance(centroids)
        stan_dev = np.std(dis, axis=0)
        average = np.average(dis, axis=0)
    plt.savefig("Voronoi Tessellations/voronoi graph {name}.png".format(name = pdf_name))
    plt.close()
    return dis, stan_dev, average


def cluster_points(points,K,pdf_name):
    if K == 0:
        return []
    if(len(points) <= K):
        K = len(points)
        print('REDO BLUR TOO HIGH')
    kmeans = KMeans(n_clusters=K,max_iter=10000)
    kmeans.fit(points)
    centers = kmeans.cluster_centers_
    pred = kmeans.fit_predict(points)
    plot_clusters(centers,pred,points,pdf_name)
    return centers

def plot_clusters(centers,pred,X,pdf_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(*zip(*X), c=pred)
    plt.grid(True)
    for center in centers:
        center = center[:2]
        plt.scatter(center[0], center[1], marker='^', c='red')
    plt.savefig("Cluster Graphs/clusters {name}.png".format(name = pdf_name))
    plt.close()
    #plt.show()


def in_box(robots, bounding_box):
    return np.logical_and(np.logical_and(bounding_box[0] <= robots[:, 0],
                                         robots[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= robots[:, 1],
                                         robots[:, 1] <= bounding_box[3]))

def voronoi(robots, bounding_box):
    #Author Cosmic from stack overflow question with link at top of this artifact
    eps = sys.float_info.epsilon
    i = in_box(robots, bounding_box)
    points_center = robots[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)
    # Compute Voronoi
    vor = Voronoi(points)
    #vor = sp.spatial.Voronoi(points)
    # Filter regions and select corresponding points
    regions = []
    points_to_filter = []  # we'll need to gather points too
    ind = np.arange(points.shape[0])
    ind = np.expand_dims(ind, axis=1)

    for i, region in enumerate(vor.regions):  # enumerate the regions
        if not region:  # nicer to skip the empty region altogether
            continue

        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not (bounding_box[0] - eps <= x and x <= bounding_box[1] + eps and
                        bounding_box[2] - eps <= y and y <= bounding_box[3] + eps):
                    flag = False
                    break
        if flag:
            regions.append(region)

            # find the point which lies inside
            points_to_filter.append(vor.points[vor.point_region == i][0, :])
    vor.filtered_points = np.array(points_to_filter)
    vor.filtered_regions = regions
    return vor

def centroid_region(vertices):
    # Author Cosmic from stack overflow question with link at top of this artifact
    A = 0

    C_x = 0

    C_y = 0
    for i in range(0, len(vertices) - 1):
        s = (vertices[i, 0] * vertices[i + 1, 1] - vertices[i + 1, 0] * vertices[i, 1])
        A = A + s
        C_x = C_x + (vertices[i, 0] + vertices[i + 1, 0]) * s
        C_y = C_y + (vertices[i, 1] + vertices[i + 1, 1]) * s
    A = 0.5 * A
    C_x = (1.0 / (6.0 * A)) * C_x
    C_y = (1.0 / (6.0 * A)) * C_y
    return np.array([[C_x, C_y]])

def plot(r,bounding_box):
    # Author Cosmic from stack overflow question with link at top of this artifact

    vor = voronoi(r, bounding_box)

    fig = plt.figure()
    ax = fig.gca()
    if vor.filtered_points.shape[0] == 0:
        return np.empty((0,1))
# Plot initial points
    ax.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], 'b.')
    #print("initial",vor.filtered_points)
# Plot ridges points
    for region in vor.filtered_regions:
        vertices = vor.vertices[region, :]
        ax.plot(vertices[:, 0], vertices[:, 1], 'go')
# Plot ridges
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        ax.plot(vertices[:, 0], vertices[:, 1], 'k-')
# Compute and plot centroids
    centroids = []
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        centroid = centroid_region(vertices)
        centroids.append(list(centroid[0, :]))
        ax.plot(centroid[:, 0], centroid[:, 1], 'r.')
    centroids = np.asarray(centroids)
    return centroids

if __name__ == '__main__':
    debug = False
    directory = 'pdf files'
    csv_file_name = 'data.txt'
    file = open(csv_file_name, mode='w', newline='')
    file.write('Name,Standard Deviation,Avg Distance\n')
    if debug:
        #find_robots('pdf files','y24a5s6d1',9)
        #find_robots('pdf files','y24a3s2d6',9)
        #find_robots('pdf files','y24a3s3d3',9)
        #find_robots('pdf files','y24a6s6d2',9)
        #find_robots('pdf files','y17a4s14d2',9)
        #find_robots('pdf files', 'y18a3s4d1', 9)
        #find_robots('pdf files', 'y17a3s9d1', 9)
        #find_robots('pdf files', 'y18a4s10d1p2', 9)
        #find_robots('pdf files', 'y19a4s3d1', 9)
        find_tessellation('pdf files', 'y23a5s6d1', 9)
    else:
        for filename in os.listdir(directory):
            f = directory + '/' + filename
            # checking if it is a file
            if os.path.isfile(f):
                name = filename[:len(filename) - 4]
                print(name)
                dis, stand_dev, average = find_tessellation(directory, name, 9)
                print(name + 'completed')
                file.write(str(name) + ',' + str(stand_dev) + ',' + str(average) + '\n')
    file.close()
