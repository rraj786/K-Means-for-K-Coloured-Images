# K-Means for K-coloured Images

This Python script implements the K-means algorithm for reducing the number of colors in an image. The K-means algorithm is a popular unsupervised machine learning technique used for clustering data. In the context of image processing, it can be used to group similar pixels together and thereby reduce the number of colors required to accurately represent an image.

## Installation

This script requires Python 3 and the following dependencies:

- ArgParse (parse user inputs)
- Matplotlib (plotting results)
- NumPy (manipulating arrays and apply mathematical operations)
- Pillow (process image data)
- Random (randomly sample data)

```bash
pip install argparse
pip install matplotlib
pip install numpy
pip install pillow
pip install random
```
    
## Usage

To use this script, follow these steps:
- Clone the repository or download the `main.py` file.
- Ensure you have installed the required dependencies.
- Run the script with the paths to the input and output images:
```bash
python main.py --imgpath eiffel.jpeg -savepathimg eiffel_processed.jpeg
```
- If you would like to modify the default inputs such as the value of K, then adapt the command below:
```bash
python main.py --imgpath eiffel.jpeg --colours 3 --iterations 30 --show True --saveconverge True --savepathimg eiffel_processed.jpeg
```

## Algorithm Overview

The K-means algorithm is a popular clustering algorithm used for partitioning a dataset into K distinct, non-overlapping clusters.

The algorithm operates as follows:

**Initialization**
   - Begin by randomly initializing K cluster centers. These cluster centers represent the initial guesses for the colors that will represent the clusters.

**Assignment Step**
   - For each pixel in the image, calculate the distance between the pixel's color and each of the K cluster centers. This distance is typically measured using the Euclidean distance in the color space (RGB in our case).
   - Assign the pixel to the cluster whose center is closest in terms of color distance.

**Update Step**
   - After all pixels have been assigned to clusters, update each cluster center by computing the mean of the colors of all pixels assigned to that cluster. This mean color becomes the new cluster center.

**Convergence Check**
   - Repeat the assignment and update steps iteratively until convergence is reached. Convergence is typically determined by measuring the change in cluster centers between iterations. If the change falls below a predefined threshold or a maximum number of iterations is reached, the algorithm terminates.

**Output**
   - Once convergence is achieved, the final cluster centers represent the reduced set of colors for the image. Each pixel in the image is then reassigned to the nearest final cluster center, resulting in the final reduced color image.

## Key Considerations

**Number of Clusters (K)**: Choosing the appropriate number of clusters is crucial. Too few clusters may result in loss of image detail, while too many clusters may not effectively reduce the number of colors. The sweetspot is somewhere between 4 and 10. 

**Initialization Method**: The performance of K-means can be sensitive to the initial choice of cluster centers. Common initialization methods include random initialization and k-means++ initialization, which selects initial centers that are well-spaced.

**Convergence Criteria**: Determining when the algorithm has converged is important. Typically, convergence is considered achieved when the change in cluster centers becomes small or when a maximum number of iterations is reached.

## Example

Here's an example of using the script to reduce the number of colors in an image (K = 3):

**Original Image**

![Original Image](eiffel.jpeg)

**Reduced Color Image (3 colors)**

![Reduced Color Image](eiffel_processed.jpeg)
## References

- https://en.wikipedia.org/wiki/K-means_clustering
