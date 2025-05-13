# ModelComprehension

## Getting started

### Setup
You need to download Python 3.* or greater to your machine, then download ModelComprehension.py. That all you need
to be able to run this program.

### How to Use

#### Running the Command Line Interface (CLI) Tool

1. Download the final release version from [here](https://github.com/Tomicgun/ModelComprehensionV2)
2. Unzip the package into a directory of your choice using any unzipping tool.
3. Open your preferred terminal application (e.g., Command Prompt, PowerShell, or any other terminal).
4. Navigate to the directory where you extracted the files.
5. Run the CLI tool using one of the following commands, depending on your system setup:

   ```bash
   python ModelComprehensionV2.py
   ```

   or

   ```bash
   python3 ModelComprehensionV2.py
   ```
   

#### Using the CLI Tool
Once the CLI is running, you’ll have full access to the system's functionality. The command accepts three arguments:
* An optional path to a configuration file (config.yml)
* A required input directory path
* A required output directory path

We recommend using the provided config.yml file or the default values. However, you are free to edit or create your own configuration file to customize the tool to your specific needs. 
The included configuration file contains detailed explanations of all parameters, allowing safe and flexible customization without breaking functionality.

That is all you need to run and use this program!

## Introduction
This project is part of a larger research effort focused on model comprehension — specifically, understanding what makes a good UML diagram.
As part of this research, we analyzed over 700 diagrams across a variety of metrics, one of the most important being model density.

Manually calculating model density for a large number of diagrams would have been completely impractical, 
which led to the idea of building a program that could measure density quickly, reliably, and consistently. 
From that idea, this project was born. We aimed to calculate the average distance between elements to determine how dense or sparse a diagram was. 
Using a combination of OCR, OpenCV, mean shift clustering, and Voronoi tessellation, we identified the center points of broad regions within each 
diagram—based on either textual or structural elements. These center points were then used to calculate distances and standard deviation, 
forming the basis of what we define as diagram density.

## Design

This section covers why we designed the program the way we did and how it works in more detail.

The **Model Comprehension** program consists of three broad steps:  
1. Finding points of interest  
2. Clustering using Voronoi tessellation  
3. Calculating the average distance and standard deviation  

---

### Finding Points of Interest

**Definition:** *Points of interest* are texts, boxes, or nodes in a diagram that hold significant semantic or contextual information for understanding the diagram.

We identify these points using two main methods:

1. OCR from the `doctr` library  
2. An OpenCV method to detect structural elements  

#### Why OCR?

OCR is excellent at detecting text. We found no better method for reliably identifying textual elements without using a neural network. OCR only occasionally failed, typically when the diagram was very large and low resolution. In those cases, we fall back on an OpenCV-based method.

#### The OpenCV Method

This method uses several techniques:

- `getStructuringElement` is used to detect the edges of shapes or elements.
- The result is then **dilated** to enhance the structures.
- `findContours` is used to identify regions of interest.
- A **threshold** is applied to remove small artifacts that may skew results.

Altogether, this process results in clean and usable outputs.

#### Why a Fallback Method? Why OpenCV?

- The fallback ensures that **some data is always extracted** from every diagram.
- In the default configuration, **both methods run**, and we use the one that produces more data.
- OpenCV is a widely used Python library with **strong community support**, **comprehensive documentation**, and is relatively **easy to work with**.

#### Optional: Clustering with Mean Shift

We optionally apply **mean shift clustering** to the data (from either method).

- It automatically determines the number of clusters.
- It smooths out artifacts and reduces the number of data points.
- Especially useful when diagrams have a large number of points (>100).

---

### Clustering Using Voronoi Tessellation

We use a custom **bounded tessellation method** based on SciPy’s Voronoi implementation.

- It finds **regions** and their **geographic center points**.
- These centers are used to calculate **average distances** and **standard deviation**.

#### Why Voronoi Tessellation?

- It partitions data into regions or clusters.
- A few isolated nodes shouldn’t suggest high density.
- Many tightly packed nodes should indicate density.
- Voronoi helps visualize clustering and spread of points of interest.

---

### Calculating Average Distance and Standard Deviation

In the final step:

1. We calculate the **average distance** from each center point to every other.
2. We do this for all points.
3. We then compute the **overall average** of these distances and their **standard deviation**.

This provides a metric for **diagram density**, or inversely:

- **High average distance** = Low density  
- **Low average distance** = High density

## Procedure

Step by step how the program runs for each diagram given to it.

* For every pdf file it creates a png file
* For every png file it then runs an optical character recognition (OCR)  method
* If the OCR returns less than 6 POIs it then runs an open cv method
* The algorithm then uses the method that returns the most POI’s.
* If the number of POI’s is larger than 100 it will use mean shift clustering to reduce the number of POI’s
* using the data from the above steps, the algorithm uses a custom voronoi method to find voronoi center points. 
* These voronoi center points are used to calculate average distance, and standard deviation of the distances.


## Authors
Thomas Marten <br>
Kritika Parajuli <br>
Jeremiah Hubbard <br>
Prof. Dr. Bastian Tenbergen <br>

## Sources
https://hpaulkeeler.com/voronoi-dirichlet-tessellations/ <br>

https://stackoverflow.com/questions/6230752/extracting-page-sizes-from-pdf-in-python <br>

https://www.geeksforgeeks.org/how-to-detect-shapes-in-images-in-python-using-opencv/ <br>

https://www.geeksforgeeks.org/python-opencv-find-center-of-contour/ <br>

https://www.geeksforgeeks.org/convert-pdf-to-image-using-python/ <br>

https://stackoverflow.com/questions/51429596/centroidal-voronoi-tessellation <br>

https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html#scipy.spatial.Voronoi <br>

https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html <br>

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html <br>

https://www.w3schools.com/python/pandas/pandas_csv.asp <br>

https://www.tutorialspoint.com/how-to-write-a-single-line-in-text-file-using-python <br>

https://www.w3schools.com/python/python_ml_k-means.asp <br>

https://stackoverflow.com/questions/37771263/detect-text-area-in-an-image-using-python-and-opencv <br>

https://pymupdf.readthedocs.io/en/latest/ <br>

https://developers.mindee.com/docs/api-documentation <br>

https://docs.python.org/3/library/argparse.html <br>

https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html <br>
