################################################################################
# Authors: Thomas Marten, Prof. Bastian Tenbergen, Kritika Parajuli            #
# This code was written for a SUNY Oswego Cyper Physical research lab          #
#                                                                              #
# This code also uses the code of a stack overflow user by the name of cosmic  #
# and his post at the link bellow:                                             #
# https://stackoverflow.com/questions/51429596/centroidal-voronoi-tessellation #
#                                                                              #
################################################################################


import numpy as np

from sklearn.cluster import MeanShift
from scipy.spatial import Voronoi
import cv2
import pymupdf
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from doctr.models import ocr_predictor
from doctr.models.predictor import OCRPredictor
from doctr.io import DocumentFile

import os
import sys

from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum
from yaml import safe_load, YAMLError
from argparse import ArgumentParser

class PoiVersion(Enum):
    OCR = 1
    OPENCV = 2
    ROLLBACK = 3

class ClusteringCriteria(Enum):
    ALWAYS = 1
    NEVER = 2
    THRESHOLD = 3

@dataclass
class Configuration:
    poi_version: PoiVersion
    rollback_threshold: int
    use_clustering: ClusteringCriteria
    clustering_threshold: int
    use_voronoi: bool
    output_intermediate_diagrams: bool
    input_dir: str
    output_dir: str

PointList = List[Tuple[int, int]]
@dataclass
class DiagramData:
    width: int
    height: int
    pois: PointList


    """
    :description: This method takes in a config file path, output directory path, and input directory path. 
    This method then enter all the config data into to Configuration class. 
    Then return this class for later use in the program.
    
    :param yaml_file_path:
    :type yaml_file_path: str 
    :param output_directory_file_path:
    :type output_directory_file_path: str
    :param input_directory_file_path:
    :type input_directory_file_path: str
    :return: Configuration
    :rtype: Class Configuration
    """
def configuration_creator(yaml_file_path, output_directory_file_path, input_directory_file_path):
    configuration = None
    try:
        with open(yaml_file_path, 'r') as config_file:
            config = safe_load(config_file)
            poi_version = PoiVersion(config['poi_version'])
            use_clustering = ClusteringCriteria(config['use_clustering'])
            configuration = Configuration(
                poi_version=poi_version,
                rollback_threshold=config['rollback_threshold'],
                clustering_threshold=config['clustering_threshold'],
                use_clustering=use_clustering,
                use_voronoi=config['use_voronoi'],
                output_intermediate_diagrams=config['output_intermediate_diagrams'],
                input_dir=input_directory_file_path,
                output_dir=output_directory_file_path
                )
    except FileNotFoundError:
        poi_version = PoiVersion(PoiVersion.ROLLBACK)
        use_clustering = ClusteringCriteria(ClusteringCriteria.THRESHOLD)
        configuration = Configuration(
            poi_version=poi_version,
            rollback_threshold=6,
            clustering_threshold=100,
            use_clustering=use_clustering,
            use_voronoi=True,
            output_intermediate_diagrams=True,
            input_dir=input_directory_file_path,
            output_dir=output_directory_file_path
        )
    except YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None


    return configuration


def run_all(config: Configuration) -> None:

    if not os.path.isdir(config.input_dir):
        raise FileNotFoundError(f'{config.input_dir} does not exist, or is not a directory')
    if not os.path.isdir(config.output_dir):
        raise FileNotFoundError(f'{config.output_dir} does not exist, or is not a directory')

    ocr_reader = None
    if config.poi_version != PoiVersion.OPENCV:
        ocr_reader = ocr_predictor(pretrained=True)

    png_directory = ensure_subdirectory(config.output_dir, 'png')
    poi_directory = ensure_subdirectory(config.output_dir, 'poi')
    cluster_directory = ensure_subdirectory(config.output_dir, 'cluster')
    voronoi_directory = ensure_subdirectory(config.output_dir, 'voronoi')
    #if config.use_clustering != ClusteringCriteria.NEVER:
    #if config.use_voronoi:

    files = [f for f in os.listdir(config.input_dir) if os.path.isfile(os.path.join(config.input_dir, f))]

    distance_distributions: List[Tuple[str, float, float]] = []

    for i, filename in enumerate(files):

        # convert pdf to png
        diagram_name = os.path.splitext(os.path.basename(filename))[0] + '.png'
        image_path = os.path.join(png_directory, diagram_name)
        if not pdf_2_image(os.path.join(config.input_dir, filename), image_path):
            print(f'failed to convert {diagram_name} to image')
            continue

        # find initial points of interest
        diagram_data = find_points_of_interest(image_path, ocr_reader, config, os.path.join(poi_directory, diagram_name))

        # cluster if appropriate
        if config.use_clustering == ClusteringCriteria.ALWAYS or config.use_clustering == ClusteringCriteria.THRESHOLD and len(diagram_data.pois) > config.clustering_threshold:
            diagram_data = cluster_points(diagram_data, config.output_intermediate_diagrams, os.path.join(cluster_directory, diagram_name))

        # voronoi
        if config.use_voronoi:
            diagram_data = find_voronoi_centroids(diagram_data, config.output_intermediate_diagrams, os.path.join(voronoi_directory, diagram_name))

        distances = find_distances(diagram_data.pois)
        distance_distributions.append((filename, np.mean(distances, axis=0), np.std(distances, axis=0)))
        print(f'finished {diagram_name} [{i + 1}/{len(files)}]')

    df = pd.DataFrame(distance_distributions, columns=['diagram', 'average', 'stddev'])
    df.to_csv(os.path.join(config.output_dir, 'output.csv'), index=False)


def ensure_subdirectory(output_root: str, subdirectory: str) -> str:
    subdirectory_path = os.path.join(output_root, subdirectory)
    if not os.path.exists(subdirectory_path):
        os.mkdir(subdirectory_path)
    return subdirectory_path


def pdf_2_image(filepath: str, outpath: str) -> bool:
    try:
        with pymupdf.open(filepath) as pdf:
            pdf.load_page(0).get_pixmap(dpi=300).save(outpath)
            return True
    except:
        return False


def find_points_of_interest(image_path: str, ocr_reader: OCRPredictor | None, config: Configuration, output_filepath: str) -> DiagramData:
    if config.poi_version != PoiVersion.OPENCV:
        image, points = poi_ocr(image_path, ocr_reader, config.output_intermediate_diagrams)

        if config.poi_version == PoiVersion.ROLLBACK and len(points) <= config.rollback_threshold:
            cv_image, cv_points = poi_pure_opencv(image_path)
            if len(cv_points) > len(points):
                image, points = cv_image, cv_points

    else:
        image, points = poi_pure_opencv(image_path)

    if config.output_intermediate_diagrams:
        cv2.imwrite(output_filepath, image)

    return DiagramData(pois=points, width=image.shape[0], height=image.shape[1])


def poi_ocr(image_path: str, ocr_reader: OCRPredictor, generate_intermediate_diagram: bool) -> Tuple[cv2.typing.MatLike | None, PointList]:
    page = ocr_reader(DocumentFile.from_images(image_path)).pages[0]

    lines = [line for block in page.blocks for line in block.lines]

    text_centers = [
        (round((line.geometry[0][0] + line.geometry[1][0]) / 2 * page.dimensions[1]),
         round((line.geometry[0][1] + line.geometry[1][1]) / 2 * page.dimensions[0]))
        for line in lines
    ]

    image = None
    if generate_intermediate_diagram:
        image = cv2.imread(image_path)
        for c in text_centers:
            cv2.circle(image, c, 10, (0, 0, 255), 5)

    return image, text_centers


def poi_pure_opencv(image_path: str) -> Tuple[cv2.typing.MatLike, PointList]:
    image = cv2.imread(image_path)

    # This method was taken from this stack exchange page https://stackoverflow.com/questions/37771263/detect-text-area-in-an-image-using-python-and-opencv
    # OP username is nathancy
    # Do old Method
    data_points = []

    # Load image, grayscale, Gaussian blur, adaptive threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # using slightly less blurr
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # block size control how large the regions are, c is a constant that is subtracted from mean, and influences how many points are found at the end
    # for block size 9 or 11 are good, for c 30 is the sweet spot more or less simply degrades results.
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if it finds only 2 contours keep the head, if not remove the head contour
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


def cluster_points(diagram_data: DiagramData, generate_intermediate_diagram: bool, output_filepath: str) -> DiagramData:
    mean_shift = MeanShift(bandwidth=100,n_jobs=4)
    mean_shift.fit(diagram_data.pois)
    centers = mean_shift.cluster_centers_

    if generate_intermediate_diagram:
        pred = mean_shift.predict(diagram_data.pois)
        plot_clusters(centers, pred, diagram_data.pois, output_filepath)

    diagram_data.pois = [(int(center[0]), int(center[1])) for center in centers]
    return diagram_data


def plot_clusters(centers: np.ndarray, pred: np.ndarray, points: PointList, output_filepath: str):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(*zip(*points), c=pred)
    plt.grid(True)
    for center in centers:
        center = center[:2]
        plt.scatter(center[0], center[1], marker='^', c='red')

    plt.savefig(output_filepath)
    plt.close()


def find_voronoi_centroids(diagram_data: DiagramData, generate_intermediate_diagram: bool, output_filepath: str) -> DiagramData:
    # Adapted from Cosmic from stack overflow question with link at top of this artifact
    vor, filtered_points, regions = voronoi(np.array(diagram_data.pois), [0, diagram_data.height, 0, diagram_data.width])
    if filtered_points.shape[0] == 0:
        diagram_data.pois = []
        return diagram_data

    centroids = []
    pois = []
    for region in regions:
        vertices = vor.vertices[region + [region[0]], :]
        centroid = centroid_region(vertices)
        centroids.append(centroid)
        pois.append((int(centroid[0, 0]), int(centroid[0, 1])))
    diagram_data.pois = pois

    if generate_intermediate_diagram:
        plot_voronoi(vor, np.asarray(centroids), filtered_points, regions, output_filepath)

    return diagram_data


def voronoi(pois: np.ndarray, bounding_box: List[int]) -> Tuple[Voronoi, np.ndarray, List[List[int]]]:
    #Author Cosmic from stack overflow question with link at top of this artifact

    eps = sys.float_info.epsilon
    i = in_box(pois, bounding_box)
    points_center = pois[i, :]
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
                if not (bounding_box[0] - eps <= x <= bounding_box[1] + eps and
                        bounding_box[2] - eps <= y <= bounding_box[3] + eps):
                    flag = False
                    break
        if flag:
            regions.append(region)

            # find the point which lies inside
            points_to_filter.append(vor.points[vor.point_region == i][0, :])
    return vor, np.array(points_to_filter), regions


def in_box(pois: np.ndarray, bounding_box: List[int]):
    return np.logical_and(np.logical_and(bounding_box[0] <= pois[:, 0],
                                         pois[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= pois[:, 1],
                                         pois[:, 1] <= bounding_box[3]))


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


def plot_voronoi(vor: Voronoi, centroids: np.ndarray, filtered_points: np.ndarray, regions: List[List[int]], output_filepath: str) -> None:

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(filtered_points[:, 0], filtered_points[:, 1], 'b.')
    for region in regions:
        vertices = vor.vertices[region, :]
        ax.plot(vertices[:, 0], vertices[:, 1], 'k-')

    for centroid in centroids:
        ax.plot(centroid[:, 0], centroid[:, 1], 'r.')

    fig.savefig(output_filepath)
    plt.close(fig)


def find_distances(points: PointList) -> List[float]:
    distances = []
    for point1 in points:
        for point2 in points:
            if point1[0] == point2[0] and point1[1] == point2[1]:
                continue
            distances.append(np.linalg.norm(np.subtract(point1, point2)))
    return distances


if __name__ == '__main__':
    parser = ArgumentParser(description='Model Comprehension V2')

    parser.add_argument("-c","--Config",required=False,type=str, help="Config file")
    parser.add_argument("-i","--Input", help="Input Directory", required=True, type=str)
    parser.add_argument("-o","--Output", help="Output Directory", required=True, type=str)
    args = parser.parse_args()

    config = args.Config
    input_dir = args.Input
    output_dir = args.Output

    config = configuration_creator(config, output_dir, input_dir)
    if(config == None):
        print("Error in configuration Setup please try again.")
        exit(1)

    run_all(config)
