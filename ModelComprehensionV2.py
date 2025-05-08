################################################################################
# Authors: Thomas Marten, Prof. Bastian Tenbergen, Kritika Parajuli,           #
# Jeremiah Hubbard                                                             #
#                                                                              #
# This code was written for a SUNY Oswego Cyper Physical research lab          #
#                                                                              #
# This code also uses the code of a stack overflow user by the name of cosmic  #
# and his post at the link bellow:                                             #
# https://stackoverflow.com/questions/51429596/centroidal-voronoi-tessellation #
#                                                                              #
################################################################################


import numpy as np
from numpy import ndarray, dtype

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

from typing import List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from yaml import safe_load, YAMLError
from argparse import ArgumentParser

class PoiVersion(Enum):
    OCR = "ocr"
    OPENCV = "opencv"
    ROLLBACK = "rollback"

class ClusteringCriteria(Enum):
    ALWAYS = "always"
    NEVER = "never"
    THRESHOLD = "threshold"

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
    open_cv_filter_threshold: int

PointList = List[Tuple[int, int]]
@dataclass
class DiagramData:
    width: int
    height: int
    pois: PointList


def configuration_creator(yaml_file_path: str, output_directory_file_path: str, input_directory_file_path: str) -> Configuration:
    """
    This method takes in a config file path, output directory path, and input directory path.
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
    configuration = None
    if yaml_file_path is not None and os.path.isfile(yaml_file_path):
        try:
            with open(yaml_file_path, 'r') as config_file:
                config = safe_load(config_file)
                poi_version = PoiVersion(config['poi_version'])
                use_clustering = ClusteringCriteria(config['clustering_criteria'])
                configuration = Configuration(
                    poi_version=poi_version,
                    rollback_threshold=config['rollback_threshold'],
                    clustering_threshold=config['clustering_threshold'],
                    use_clustering=use_clustering,
                    use_voronoi=config['use_voronoi'],
                    output_intermediate_diagrams=config['output_intermediate_diagrams'],
                    open_cv_filter_threshold=config['open_cv_filter_threshold'],
                    input_dir=input_directory_file_path,
                    output_dir=output_directory_file_path
                    )
        except (YAMLError) as exc:
            print(f"Error parsing YAML file: {exc}")
            return None  
        except KeyError as exc: 
            print(f'Missing configuration key {exc}')
            return None
    else:
        poi_version = PoiVersion(PoiVersion.ROLLBACK)
        use_clustering = ClusteringCriteria(ClusteringCriteria.THRESHOLD)
        configuration = Configuration(
            poi_version=poi_version,
            rollback_threshold=6,
            clustering_threshold=100,
            use_clustering=use_clustering,
            use_voronoi=True,
            output_intermediate_diagrams=True,
            open_cv_filter_threshold=1000,
            input_dir=input_directory_file_path,
            output_dir=output_directory_file_path
        )
    return configuration


def run_all(config: Configuration) -> None:
    """
    This method takes in a configuration object, it then makes sure the necessary directory are created.
    then for every file it runs the configured methods OCR,open_cv,mean shift clustering, and voronoi clustering. It then
    finds the distances between all every point to every point. Then saves all this data to a csv file.
    :param config: The configuration object
    :type config: Configuration
    :rtype: None
    :raise FileNotFoundError: No input or output file directory provided by user
    """

    """create an input or output directory if none is found"""
    if not os.path.isdir(config.input_dir):
        raise FileNotFoundError(f'{config.input_dir} does not exist, or is not a directory')
    if not os.path.isdir(config.output_dir):
        raise FileNotFoundError(f'{config.output_dir} does not exist, or is not a directory')

    """Configure the OCR"""
    ocr_reader = None
    if config.poi_version != PoiVersion.OPENCV:
        ocr_reader = ocr_predictor(pretrained=True)

    """Create the Necessary file structer"""
    png_directory = ensure_subdirectory(config.output_dir, 'png')
    poi_directory = ensure_subdirectory(config.output_dir, 'poi')
    cluster_directory = ensure_subdirectory(config.output_dir, 'cluster')
    voronoi_directory = ensure_subdirectory(config.output_dir, 'voronoi')

    """Append the file path for each file name"""
    files = [f for f in os.listdir(config.input_dir) if os.path.isfile(os.path.join(config.input_dir, f))]

    """configure the distance distribution tuple list"""
    distance_distributions: List[Tuple[str, float, float]] = []

    for i, filename in enumerate(files):
        
        """convert pdf to png"""
        diagram_name = os.path.splitext(os.path.basename(filename))[0] + '.png'
        image_path = os.path.join(png_directory, diagram_name)
        if not pdf_2_image(os.path.join(config.input_dir, filename), image_path):
            print(f'failed to convert {diagram_name} to image')
            continue

        """find initial points of interest"""
        diagram_data = find_points_of_interest(image_path, ocr_reader, config, os.path.join(poi_directory, diagram_name))

        """cluster if appropriate"""
        if config.use_clustering == ClusteringCriteria.ALWAYS or config.use_clustering == ClusteringCriteria.THRESHOLD and len(diagram_data.pois) > config.clustering_threshold:
            diagram_data = cluster_points(diagram_data, config.output_intermediate_diagrams, os.path.join(cluster_directory, diagram_name))

        """Voronoi clustering"""
        if config.use_voronoi:
            diagram_data = find_voronoi_centroids(diagram_data, config.output_intermediate_diagrams, os.path.join(voronoi_directory, diagram_name))

        """find the distances, and the distance to the tuple list"""
        distances = find_distances(diagram_data.pois)
        distance_distributions.append((filename, np.mean(distances, axis=0), np.std(distances, axis=0)))
        print(f'finished {diagram_name} [{i + 1}/{len(files)}]')

    """covert the all the distance distributions tuples to a data frame"""
    df = pd.DataFrame(distance_distributions, columns=['diagram', 'average', 'stddev'])

    """Save the data frame to CSV"""
    df.to_csv(os.path.join(config.output_dir, 'output.csv'), index=False)



def ensure_subdirectory(output_root: str, subdirectory: str) -> str:
    """
    This method ensures that all subdirectories are present in the output directory.
    :param output_root:
    :param subdirectory:
    :return: str
    """
    subdirectory_path = os.path.join(output_root, subdirectory)
    if not os.path.exists(subdirectory_path):
        os.mkdir(subdirectory_path)
    return subdirectory_path


def pdf_2_image(filepath: str, outpath: str) -> bool:
    """
    Converts the given PDF at the filepath to a png image and saves
    it to the provided subdirectory in the output directory.
    :param filepath:
    :type filepath: str
    :param outpath:
    :type outpath: str
    :return: bool
    """
    try:
        with pymupdf.open(filepath) as pdf:
            pdf.load_page(0).get_pixmap(dpi=300).save(outpath)
            return True
    except:
        return False


def find_points_of_interest(image_path: str, ocr_reader: OCRPredictor | None, config: Configuration, output_filepath: str) -> DiagramData:
    """
    This method is where the actually OCR and Open_cv methods are runed to find the points of intrest.
    If OCR or Rollback are configured it will use ether method, if nothing is specified open_cv will be
    used as default. If rollback is configured and OCR gets less than the threshold we will use the
    method that produces the most data.
    :param image_path:
    :type image_path: str
    :param ocr_reader:
    :type ocr_reader: OCRPredictor | None
    :param config:
    :type config: Configuration
    :param output_filepath:
    :type output_filepath: str
    :return: DiagramData
    :rtype DiagramData: DiagramData object
    """
    if config.poi_version != PoiVersion.OPENCV:
        image, points = poi_ocr(image_path, ocr_reader, config.output_intermediate_diagrams)

        if config.poi_version == PoiVersion.ROLLBACK and len(points) <= config.rollback_threshold:
            cv_image, cv_points = poi_pure_opencv(image_path,config)
            if len(cv_points) > len(points):
                image, points = cv_image, cv_points

    else:
        image, points = poi_pure_opencv(image_path,config)

    """outputting intermediate diagrams if needed"""
    if config.output_intermediate_diagrams:
        cv2.imwrite(output_filepath, image)

    """Returning the data points found from OCR or Open_cv"""
    return DiagramData(pois=points, width=image.shape[0], height=image.shape[1])
    

def poi_ocr(image_path: str, ocr_reader: OCRPredictor, generate_intermediate_diagram: bool) -> Tuple[cv2.typing.MatLike | None, PointList]:
    """
    This method is where doctr OCR method is used to find center points of text. It also can generate
    an intermediate diagram if configured too showing what OCR found.
    :param image_path:
    :param ocr_reader:
    :param generate_intermediate_diagram:
    :returns: image | none and a list of text center points
    :rtype image: MatLike | None
    :rtype points: (int,int)
    """
    page = ocr_reader(DocumentFile.from_images(image_path)).pages[0]

    """Getting all the text lines from the png in nested list comprehension"""
    lines = [line for block in page.blocks for line in block.lines]

    """Find the center points of the lines/text again using list comprehension"""
    text_centers = [
        (round((line.geometry[0][0] + line.geometry[1][0]) / 2 * page.dimensions[1]),
         round((line.geometry[0][1] + line.geometry[1][1]) / 2 * page.dimensions[0]))
        for line in lines
    ]

    """Generate the image if necessary"""
    image = None
    if generate_intermediate_diagram:
        image = cv2.imread(image_path)
        for c in text_centers:
            cv2.circle(image, c, 10, (0, 0, 255), 5)

    return image, text_centers


def poi_pure_opencv(image_path: str, config: Configuration) -> Tuple[cv2.typing.MatLike, PointList]:
    """
    This method is for actually finding the center points of POIs using OpenCV
    :param image_path:
    :type image_path: str
    :returns: image | none and a list of text center points
    :rtype image: MatLike | None
    :rtype points: (int,int)
    """
    image = cv2.imread(image_path)

    """
    This method was taken from this stack exchange page https://stackoverflow.com/questions/37771263/detect-text-area-in-an-image-using-python-and-opencv
    OP username is nathancy
    """
    data_points = []

    """grayscale, Gaussian blur, adaptive threshold"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    """
    block size control how large the regions are, c is a constant that is subtracted from mean, and influences how many points are found at the end
    for block size 9 or 11 are good, for c 30 is the sweet spot more or less simply degrades results.
    """
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 30)

    """Dilate to combine adjacent text contours"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    """Find contours, highlight text areas, and extract POIs"""
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    """if it finds only 2 contours keep the head, if not remvoe the head contour"""
    contours = contours[0] if len(contours) == 2 else contours[1]

    """
    Do the drawing and saving of points
    Filter out any contour areas less than the threshold of 1000
    higher or lower will cause more or less points to be found
    """
    for c in contours:
        area = cv2.contourArea(c)
        if area > config.open_cv_filter_threshold:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
            cv2.circle(image, (x + w // 2, y + h // 2), 10, (255, 0, 0), 2)
            data_points.append((x, y))

    return image, data_points

"""Give the Cluster points method more than 4 colors"""
colors = [
    (255, 128, 128),
    (255, 191, 128),
    (255, 255, 128),
    (191, 255, 128),
    (128, 255, 128),
    (128, 255, 191),
    (128, 255, 255),
    (128, 191, 255),
    (128, 128, 255),
    (191, 128, 255),
]
def cluster_points(diagram_data: DiagramData, generate_intermediate_diagram: bool, output_filepath: str) -> DiagramData:
    """
    This method takes in the diagram data from the run all method and
    uses mean shift clustering to find a new set of diagram points
    based of the center points of the cluster mean shift found.

    :param diagram_data:
    :type diagram_data: DiagramData
    :param generate_intermediate_diagram:
    :type generate_intermediate_diagram: bool
    :param output_filepath:
    :type output_filepath: str
    :return: Clustered diagram data
    :rtype: DiagramData
    """
    mean_shift = MeanShift(bandwidth=100,n_jobs=4)
    mean_shift.fit(diagram_data.pois)
    centers = mean_shift.cluster_centers_

    if generate_intermediate_diagram:
        pred = mean_shift.predict(diagram_data.pois)
        plot_clusters(centers, pred, diagram_data.pois, output_filepath)

    diagram_data.pois = [(int(center[0]), int(center[1])) for center in centers]
    return diagram_data


def plot_clusters(centers: np.ndarray, pred: np.ndarray, points: PointList, output_filepath: str):
    """
    This is a helper method for plotting the resulting cluster from the cluster_points method
    :param centers:
    :type centers: np.ndarray
    :param pred:
    :type pred: np.ndarray
    :param points:
    :type points: PointList
    :param output_filepath:
    :type output_filepath: str
    """
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
    """
    Adapted from Cosmic from stack overflow question with link at top of this artifact
    This method is a helper method that will find the bounded voronoi regions and there
    associated center points. This method will then also plot the resulting voronoi regions
    and centers if necessary
    :param diagram_data:
    :type diagram_data: DiagramData
    :param generate_intermediate_diagram:
    :type generate_intermediate_diagram: bool
    :param output_filepath:
    :type output_filepath: str
    :return: diagram_data
    :rtype: DiagramData
    """
    vor = voronoi(np.array(diagram_data.pois), [0, diagram_data.width, 0, diagram_data.height])

    centroids = []
    pois = []
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        centroid = centroid_region(vertices)
        centroids.append(centroid)
        pois.append(list(centroid[0, :]))
    diagram_data.pois = pois

    if generate_intermediate_diagram:
        diagram_data.intermediate_diagram = plot_voronoi(vor, np.array(centroids), output_filepath)

    return diagram_data


def voronoi(pois: np.ndarray, bounding_box: List[int]) -> Voronoi:
    """
    Author Cosmic from stack overflow question with link at top of this artifact
    This method creates a bounded voronoi and then finds the geometric center points
    of the resulting regions
    :param pois:
    :type pois: np.ndarray
    :param bounding_box:
    :type bounding_box: List[int]
    :return: voronoi object containing all regions and center points
    :rtype: Voronoi
    """

    """Used as a margin of error"""
    eps = sys.float_info.epsilon

    """Filters out points not inside the bounding box"""
    i = in_box(pois, bounding_box)

    """
    Creation of the bounding box using reflection the method reflects all the data points over
    the bounding box, which when given to the scipy voronoi method returns a result as if the
    scipy method had a bounding box. The bounding box is important to find the center points of the
    resulting regions
    """
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
    """Computer the voronoi center points"""
    vor = Voronoi(points)

    """
    Filter any region with vertexes outside the bounding box, basically removes all the data reflected over the bounding box 
    and select corresponding points for the unfiltered regions
    """
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
    """Convert to np array and save the center points and filtered regions inside the vor object"""
    vor.filtered_points = np.array(points_to_filter)
    vor.filtered_regions = regions
    return vor


def in_box(pois: np.ndarray, bounding_box: List[int]) -> ndarray[tuple[int, ...], dtype[Any]]:
    """
    Author Cosmic from stack overflow question with link at top of this artifact
    This method using np logical method asks if a point is inside the bounding box
    and if the points is outside the bounding box it is removed from the set of POIs
    :param pois:
    :type pois: np.ndarray
    :param bounding_box:
    :type bounding_box: List[int]
    :return: any points inside the bounding box
    :rtype: ndarray[tuple[int, ...], dtype[Any]]
    """
    return np.logical_and(np.logical_and(bounding_box[0] <= pois[:, 0],
                                         pois[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= pois[:, 1],
                                         pois[:, 1] <= bounding_box[3]))


def centroid_region(vertices):
    """
    Author Cosmic from stack overflow question with link at top of this artifact
    This method finds and creates the actual regions of the voronoi using the vertexes from the scipy voronoi method
    :param vertices:
    :return: np array that makes up a region
    """
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


def plot_voronoi(vor: Voronoi, centroids: np.ndarray, output_filepath: str) -> None:
    """
    Author Cosmic from stack overflow question with link at top of this artifact
    This method plots the regions and center points of resulting data from the voronoi method
    :param vor: The Voronoi object
    :type vor: Voronoi
    :param centroids: The center points of the voronoi regions
    :type centroids: np.ndarray
    :param output_filepath: Where to save the resulting intermediate diagram
    :type output_filepath: str
    :return: None
    """
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], 'b.')
    for region in vor.filtered_regions:
        vertices = vor.vertices[region, :]
        ax.plot(vertices[:, 0], vertices[:, 1], 'k-')

    for centroid in centroids:
        ax.plot(centroid[:, 0], centroid[:, 1], 'r.')

    fig.savefig(output_filepath)
    plt.close(fig)
    

def find_distances(points: PointList) -> List[float]:
    """
    This method find the distance between a point and every other point. It will do this
    for every point in the points list
    :param points:
    :type points: PointList
    :return: list of distances between a point and every other point for every point
    :rtype: List[float]
    """
    distances = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            distances.append(np.linalg.norm(np.subtract(points[i], points[j])))
    return distances


if __name__ == '__main__':
    """
    The actual CLI implementation is held in the main method and is where we get the
    CLI arguments and save it to the configuration object.
    """
    parser = ArgumentParser(description='Model Comprehension V2')

    parser.add_argument("-c","--Config",required=False,type=str, help="The Configuration file, please reference the example config file "
                                                                      "at https://github.com/tenbergen/ModelComprehension "
                                                                      "for further understanding how to make your own configuration file")
    parser.add_argument("-i","--Input", help="The Input Directory", required=True, type=str)
    parser.add_argument("-o","--Output", help="The Output Directory", required=True, type=str)
    args = parser.parse_args()

    config = args.Config
    input_dir = args.Input
    output_dir = args.Output

    config = configuration_creator(config, output_dir, input_dir)
    if(config == None):
        print("Error in configuration Setup please try again.")
        exit(1)

    run_all(config)
