################################################################################
# Authors: Thomas Marten, Prof. Bastian Tenbergen, Kritika Parajuli            #
# This code was written for a SUNY Oswego Cyper Physical research lab          #
#                                                                              #
# This code also uses the code of a stack overflow user by the name of cosmic  #
# and his post at the link bellow:                                             #
# https://stackoverflow.com/questions/51429596/centroidal-voronoi-tessellation #
#                                                                              #
################################################################################

from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import Voronoi
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import sys
import cv2
import pymupdf
import pandas as pd
import os
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import matplotlib.pyplot as plt

def pdf_2_image(filepath, outdir):
    try:
        with pymupdf.open(filepath) as pdf:
            png_filename = os.path.splitext(os.path.basename(filepath))[0] + '.png'
            png_filepath = os.path.join(outdir, png_filename)
            pdf.load_page(0).get_pixmap(dpi=300).save(png_filepath)
            return png_filepath
    except:
        print(f'failed to convert {filepath} to image')
        return None

def distance(points):
    distances = []
    for point1 in points:
        for point2 in points:
            if point1[0] != point2[0] or point1[1] != point2[1]:
                distances.append(np.linalg.norm(point1-point2))
    return distances

#median is 25 IQR is 23 for the new contour method
# Any text value, on line 55, less than 10 should be put through the old method
# Then check the count and if it is still less than 10, return message to user
# that image could not be processed, and ask the user to improve the image quality
# Any text value above X (Probably a 100) should probably be clustered (if deemed necessary)
def find_contour(png_filepath, pdf_name, ocr_reader, root_dir, allow_fallback=True):
    ocr_result = optical_character_recognition(png_filepath, pdf_name, ocr_reader)
    if ocr_result is None:
        return
    image, data = ocr_result

    if allow_fallback and len(data) <= 6:
        opencv_image, opencv_data = pure_open_cv_method(png_filepath, pdf_name)
        if len(data) < len(opencv_data):
            data = opencv_data
            image = opencv_image
            print(f'old method better for {pdf_name}')
            
    cv2.imwrite(f'{root_dir}/contours/{pdf_name}.png', image)
    print(f'{pdf_name} done; {len(data)} points found')
    # return cluster_points(data_points,K,pdf_name)

def find_tessellation(png_filepath, pdf_name, ocr_predictor, allow_fallback, root_dir):
    # df = pd.read_csv('Name to Nodes.csv')
    # df.columns = ['Name','Nodes']
    # df = df[df['Name'] == pdf_name]
    # if df.shape[0] == 0:
    #     print('PDF not in table')
    #     return 0,0,0
    # K = df.values.flatten().tolist()[1]
    tesselation = find_contour(png_filepath,pdf_name,ocr_predictor, root_dir, allow_fallback)
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
    plt.savefig(f'{root_dir}/voronoi/{pdf_name}.png')
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

def pure_open_cv_method(png_filepath,pdf_name):
    image = cv2.imread(png_filepath)

    # This method was taken from this stack exchange page https://stackoverflow.com/questions/37771263/detect-text-area-in-an-image-using-python-and-opencv
    # OP username is nathancy
    # Do old Method
    data_points = []

    # Load image, grayscale, Gaussian blur, adaptive threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # using slightly less blurr
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # block size control how large the regions are, c is a constant that is subtracted from mean, and infulences how many points are foind at the end
    # for block size 9 or 11 are good, for c 30 is the sweet spot more or less simply degrades results.
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if it finds only 2 contours keep the head, if not remvoe the head contour
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Do the drawing and saving of points
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
            cv2.circle(image, (x + w // 2, y + h // 2), 10, (255, 0, 0), 2)
            data_points.append((x, y))

    return image, data_points

def optical_character_recognition(png_filepath,pdf_name,predictor):
    image = cv2.imread(png_filepath)

    result = predictor(DocumentFile.from_images(png_filepath))

    lines = [line for block in result.pages[0].blocks for line in block.lines]
    dims = result.pages[0].dimensions

    data_points = [
        (int(round((line.geometry[0][0] + line.geometry[1][0]) / 2 * dims[1])),
         int(round((line.geometry[0][1] + line.geometry[1][1]) / 2 * dims[0])))
        for line in lines
    ]

    for c in data_points:
        cv2.circle(image, c, 10, (0, 0, 255), 5)

    return image, data_points

def normalize_points(image, data_points, scaling_factor):
    scaled_points = [(x / scaling_factor, y / scaling_factor) for (x,y) in data_points]
    return image, scaled_points


def normalize_points_by_resolution(image, data_points):
    # 1 / average dimension * 1000 to bring data roughly into [0,1000]
    scaling_factor = 2000 / (image.shape[0] + image.shape[2])
    return normalize_points(image, data_points, scaling_factor)
    
def normalize_points_by_font_size(lines, page_dimensions, image, data_points):
    heights = np.array([(line.geometry[1][1] - line.geometry[0][1]) * page_dimensions[0] for line in lines])

    # use kernel density estimation to find the mode (most common) font size
    kde = gaussian_kde(heights, bw_method=0.3)
    x_grid = np.linspace(min(heights) - 1, max(heights) + 1, 1000)
    kde_values = kde(x_grid)
    peaks, _ = find_peaks(kde_values)
    if len(peaks) > 0:
        font_size = x_grid[peaks[np.argmax(kde_values[peaks])]]
        print(f'estimated mode {font_size:.2f}')
    else:
        # fall back to average
        font_size = np.mean(heights)
        print(f'fell back to average: f{font_size:.2f}')
        
    # scale font size to 12pt
    return normalize_points(image, data_points, 12 / font_size)


if __name__ == '__main__':
    debug = False
    allow_old_contour_fallback = False
    directory = 'diagrams/raw'
    predictor = ocr_predictor(pretrained=True)
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        if not os.path.isfile(file):
            continue
        png_filepath = pdf_2_image(file, 'diagrams/png')
        if png_filepath is None:
            continue
        stem = os.path.splitext(os.path.basename(filename))[0]
        if debug:
            find_contour(png_filepath, stem, predictor, 'diagrams', allow_old_contour_fallback)
        else:
            dis, stand_dev, average = find_tessellation(png_filepath, stem, predictor, allow_old_contour_fallback, 'diagrams')
            print(dis, stand_dev, average)