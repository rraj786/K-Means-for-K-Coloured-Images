'''
    The following script encompasses a k-means algorithm that can be applied
    to images to reduce the number of colours present. k >= 1
    Author: Rohit Rajagopal
'''


import numpy as np
import pandas as pd
import random
from PIL import Image
import argparse


def SelectKRandomPoints(array, k):

    """
        Randomly sample k coordinates based on input dataset.
        Inputs:
            - array: A 3D image array
            - k: Number of points to sample
        Outputs:
            - coords: A 2D array of randomly selected pixel coordinates
    """

    # Find size of input array
    rows, cols = len(array), len(array[0])

    # Sample points
    y, x = random.sample(range(rows), k), random.sample(range(cols), k)

    coords = np.vstack((y, x)).T

    return coords
    

def GetRGBValuesForPoints(array, points):

    """
        Extract RGB pixel values for a defined list of image coordinates.
        Inputs:
            - array: A 3D image array
            - points: A 2D array of image coordinates
        Outputs:
            - rgb_vals: A 3D array of RGB pixel values
    """

    pixels = array[points[0][0], points[0][1]]
    k = len(pixels)

    # Array slicing to extract RGB values
    for i in range(1, len(points)):
        pixels = np.vstack((pixels, array[points[i][0], points[i][1]]))

    rgb_vals = np.reshape(pixels, (k, 1, 3))

    return rgb_vals


def SquaredDistance(vec1, vec2):

    """
        Determines the distance between two points in 3D space.
        Inputs:
            - vec1: A single RGB point
            - vec2: Another RGB point
        Outputs:
            - dist: Distance between the two RGB points
    """
        
    # Find Euclidean distance 
    dist = (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2 + (vec1[2] - vec2[2])**2

    return dist


def AssignToClusters(array, pixels):
    
    """
        Group each pixel in an image to a designated cluster based on the mean it is closest to.
        Inputs:
            - array: A 3D image array
            - pixels: A 3D array of pixel values for each of the means
        Outputs:
            - clusters: A 2D array containing the cluster each pixel belongs to
    """

    clusters = np.ones([len(array), len(array[0])])

    # Iterate through each of the array points to find closest cluster
    for i in range(len(array)):
        for j in range(len(array[0])): 
            min = np.inf

            # Find pixel closest to point
            for count, pixel in enumerate(pixels):
                temp = SquaredDistance(pixel[0].astype(float), array[i][j].astype(float))
                if temp < min:
                    min = temp
                    cluster = count + 1
        
            # Assign cluster
            clusters[i][j] = cluster

    return clusters


def UpdateMeans(array, k, clusters):

    """
        Determines the mean values for each cluster.
        Inputs:
            - array: A 3D image array
            - k: Number of clusters
            - clusters: A 2D array containing the cluster each pixel belongs to
        Outputs:
            - means: A 3D array containing the means for each cluster and RGB layer
    """

    means = np.zeros([k, 1, 3])

    # Extract RGB values for each cluster
    for i in range(k):
        coords = np.where(clusters == i + 1)
        rgb = array[coords[0], coords[1]]

        # Calculate means for each colour 
        means[i][0][0] = np.mean(rgb[:, 0])     # Red
        means[i][0][1] = np.mean(rgb[:, 1])     # Green
        means[i][0][2] = np.mean(rgb[:, 2])     # Blue

    return means


def KMeansRGB(array, means, iterations):

    """
        Assign each point to a cluster and update as necessary.
        Inputs:
            - array: A 3D image array
            - means: A 3D array containing the means to initialise the algorithm
            - iterations: Maximum number of iterations to update clusters
        Outputs:
            - clusters: A 2D array containing the cluster each pixel belongs to
            - new_means: A 3D array containing the updated means for each cluster and RGB layer
    """

    # Update means until convergence or max iterations reached
    for i in range(iterations):
        clusters = AssignToClusters(array, means)
        new_means = UpdateMeans(array, len(means), clusters)

        # Iterations check
        if i == iterations - 1:
            print("Maximum number of iterations reached before convergence was achieved")
            break

        # Convergence check
        if (np.linalg.norm(means) - np.linalg.norm(new_means)) < tol:
            break

        means = new_means

    return clusters, new_means


def CreateKColourImage(clusters, means):

    """
        Create a k-colour image given its clusters.
        Inputs:
            - clusters: A 2D array containing the cluster each pixel belongs to
            - means: A 3D array containing the updated means for each cluster and RGB layer
        Outputs:
            - proc_image: A processed 3D image array
    """
    
    array = np.zeros([len(clusters), len(clusters[0]), 3])

    # Assign means to corresponding clusters to form 3D array
    for i in range(len(means)):
        coords = np.where(clusters == i + 1)
        array[coords[0], coords[1], 0] = means[i][0][0]     # Red
        array[coords[0], coords[1], 1] = means[i][0][1]     # Green
        array[coords[0], coords[1], 2] = means[i][0][2]     # Blue

    proc_image = array.astype(np.int8)

    return proc_image


if __name__ == "__main__":
    
    # User inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='eiffel.jpeg', help='Path to image for processing')
    parser.add_argument('--colours', type=int, default=3, help='Number of colours to represent image')
    parser.add_argument('--iterations', type=int, default=30, help='Number of iterations for updating clusters')
    parser.add_argument('--show', type=bool, default=True, help='Show original and processed images')
    parser.add_argument('--savepath', type=str, default='eiffel_processed.jpeg', help='Path to save processed image')
    args = parser.parse_args()

    # Initialise image
    tol = 1e-7
    img = Image.open(args.imgpath)
    array = np.asarray(img)

    # Call functions
    points = SelectKRandomPoints(array, args.colours)
    pixels = GetRGBValuesForPoints(array, points)
    clusters, means = KMeansRGB(array, pixels, args.iterations)   
    array = CreateKColourImage(clusters, means)

    k_img = Image.fromarray(array, "RGB")
    
    # Show images
    if args.show:
        dst = Image.new('RGB', (img.width + k_img.width + 32, img.height))
        dst.paste(img, (0, 0))
        dst.paste(k_img, (img.width + 32, 0))

        dst.show()

    # Save processed image
    k_img.save(args.savepath)
    
