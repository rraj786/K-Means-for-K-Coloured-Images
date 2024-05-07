'''
    The following script encompasses a k-means algorithm that can be applied
    to images to reduce the number of colours present. k >= 1
    Author: Rohit Rajagopal
'''

import numpy as np
import pandas as pd
import random
from PIL import Image


def SelectKRandomPoints(array, k):

    # Find size of input array
    rows, cols = len(array), len(array[0])

    # Sample points
    y, x = random.sample(range(rows), k), random.sample(range(cols), k)

    coords = np.vstack((y, x)).T

    return coords
    

def GetRGBValuesForPoints(array, points):

    pixels = array[points[0][0], points[0][1]]

    # Array slicing to extract RGB values
    for i in range(1, len(points)):
        pixels = np.vstack((pixels, array[points[i][0], points[i][1]]))

    rgb_vals = np.reshape(pixels, (k, 1, 3))

    return rgb_vals


def SquaredDistance(vec1, vec2):
     
    # Find Euclidean distance 
    dist = (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2 + (vec1[2] - vec2[2])**2

    return dist


def AssignToClusters(array, pixels):
    
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

    # Enter inputs
    tol = 1e-7
    img = Image.open("clocktower.jpg")
    img.show()
    k = 1
    array = np.asarray(img)
    iterations = 30

    # Call functions
    points = SelectKRandomPoints(array, k)
    pixels = GetRGBValuesForPoints(array, points)
    clusters, means = KMeansRGB(array, pixels, iterations)   
    array = CreateKColourImage(clusters, means)

    k_img = Image.fromarray(array, "RGB")
    k_img.show()