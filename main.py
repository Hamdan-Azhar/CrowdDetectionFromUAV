import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# Fast 9 method
def original_fast(gray_scale_image, threshold=30):
    """
    Detects corners in a grayscale image using the Fast 9 method.

    Args:
        gray_scale_image (numpy.ndarray): The grayscale image to detect corners in.
        threshold (int, optional): The threshold value for the Fast 9 method. Defaults to 30.

    Returns:
        numpy.ndarray: An array of corner coordinates.

    """
    fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=True)
    corners = fast.detect(gray_scale_image, None)
    corners = np.array([kp.pt for kp in corners])
    # plt.title("Fast 9 method")
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(gray_scale_image, cv2.COLOR_GRAY2BGR))
    plt.scatter(corners[:, 0], corners[:, 1], c='blue', s=2)
    plt.title("FAST 9 method")
    plt.show()

    return corners

# Custom Fast method (base paper method)
def custom_fast(gray_scale_image, threshold=40, n=9):
    """
    Detects corners in a grayscale image using a custom FAST algorithm.

    Args:
        gray_scale_image (ndarray): The grayscale image to detect corners in.
        threshold (int, optional): The threshold value for corner detection. Defaults to 40.
        n (int, optional): The number of pixels to consider for corner detection. Defaults to 9.

    Returns:
        ndarray: An array of corner coordinates in the format [[x1, y1], [x2, y2], ...].

    This function implements a custom version of the FAST (Features from Accelerated Segment Test) algorithm for corner detection.
    It iterates over each pixel in the grayscale image and checks the intensity values of the surrounding pixels to determine if it is a corner.
    The algorithm considers a circle of pixels centered at the current pixel and selects the nth brightest and nth dimmest pixels.
    The difference between the chosen pixel and the center pixel is calculated, and if it is within the threshold range, the corner is considered.
    The function returns an array of corner coordinates in the format [[x1, y1], [x2, y2], ...].

    Note:
        - The image should be a grayscale image.
        - The threshold value determines the sensitivity of the corner detection.
        - The value of n determines the number of pixels to consider for corner detection.
    """
    corners = []
    rows, cols = gray_scale_image.shape
    for y in range(3, rows-3):
        for x in range(3, cols-3):
            center_pixel = gray_scale_image[y, x]
            # print(center_pixel)
            circle_pixels = [
                gray_scale_image[y-3, x], gray_scale_image[y-3, x+1], gray_scale_image[y-2, x+2], gray_scale_image[y-1, x+3],
                gray_scale_image[y, x+3], gray_scale_image[y+1, x+3], gray_scale_image[y+2, x+2], gray_scale_image[y+3, x+1],
                gray_scale_image[y+3, x], gray_scale_image[y+3, x-1], gray_scale_image[y+2, x-2], gray_scale_image[y+1, x-3],
                gray_scale_image[y, x-3], gray_scale_image[y-1, x-3], gray_scale_image[y-2, x-2], gray_scale_image[y-3, x-1]
            ]
            ascending_sorted_pixels = sorted(circle_pixels)
            chosen_pixel = ascending_sorted_pixels[n-1]  # For FAST-9, n=9
            difference = np.subtract(chosen_pixel, center_pixel, dtype=np.int16)
            if abs(difference) <= -threshold:
                corners.append([x, y])
            else:
                descending_sorted_pixels = sorted(circle_pixels, reverse=True)
                chosen_pixel = descending_sorted_pixels[n-1]
                difference = np.subtract(chosen_pixel, center_pixel, dtype=np.int16)
                # print(chosen_pixel, center_pixel, difference)
                if abs(difference) >= threshold:
                    corners.append([x, y])

    corners = np.array(corners)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(gray_scale_image, cv2.COLOR_GRAY2BGR))
    plt.scatter(corners[:, 0], corners[:, 1], c='blue', s=2)
    plt.title("Base paper method")
    plt.show()
    return corners

# Remove the features that do not belong to crowd in gray_scale_image
def crowd_features_filteration(gray_scale_image, corner_points, radius=100, num_sectors=4, min_points_per_sector=50, no_of_clusters=1, method_name=""):
    """
    Filter the crowd features in a grayscale image based on their proximity to center points.

    Parameters:
        gray_scale_image (numpy.ndarray): The grayscale image in which to filter the features.
        corner_points (numpy.ndarray): The coordinates of the corner points in the image.
        radius (int, optional): The maximum distance from the center point for a feature to be considered. Defaults to 100.
        num_sectors (int, optional): The number of sectors to divide the image into for filtering. Defaults to 4.
        min_points_per_sector (int, optional): The minimum number of points required in a sector for it to be considered. Defaults to 50.
        no_of_clusters (int, optional): The number of clusters to use in the K-means clustering. Defaults to 1.

    Returns:
        numpy.ndarray: An array of filtered corner points.

    This function filters the crowd features in a grayscale image based on their proximity to center points. It first uses K-means clustering to find the center points in the image. Then, for each center point, it divides the image into sectors and filters the corner points based on their distance and angle from the center point. The filtered points are then plotted and returned as a numpy array.
    """
    # Define the K-means clustering procedure to find the center point
    kmeans = KMeans(n_clusters=no_of_clusters).fit(corner_points)

    # get center points from kmeans
    center_points = []
    for center in kmeans.cluster_centers_:
        center_points.append(center)

    # Plot the extracted features and the center point
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(gray_scale_image, cv2.COLOR_GRAY2BGR))

    for center_point in center_points:
        plt.scatter(center_point[0], center_point[1], c='green', s=10)

    theta = np.linspace(0, 2 * np.pi, num_sectors + 1)
    filtered_points = []
    
    for center_point in center_points:
        for i in range(num_sectors):
            start_angle = theta[i]
            end_angle = theta[i+1]
            sector_points = []

            for point in corner_points:
                if not np.all(point == center_point):
                    angle = np.arctan2(point[1] - center_point[1], point[0] - center_point[0])
                    angle = angle if angle >= 0 else angle + np.pi
                    distance = np.linalg.norm(point - center_point)
                    if start_angle <= angle <= end_angle and distance <= radius:
                        sector_points.append(point)

            if len(sector_points) > 0:
                if len(sector_points) >= min_points_per_sector:
                    sector_points = np.array(sector_points)
                    plt.scatter(sector_points[:, 0], sector_points[:, 1], c='blue', s=2)
                    filtered_points.extend(sector_points.tolist())
                else:
                    sector_points = np.array(sector_points)
                    plt.scatter(sector_points[:, 0], sector_points[:, 1], c='red', s=2)

    plt.title(f"crowd feature extraction for {method_name}")
    plt.show()
    # print(filtered_points)
    return np.array(filtered_points)

# density estimation using 2D gaussian kernel
def estimate_density(points, bandwidth=1.0, method_name=""):
    """
    Estimate the density of a set of points using a 2D Gaussian kernel density estimation.

    Parameters:
        points (numpy.ndarray): An array of shape (n_points, 2) containing the coordinates of the points.
        bandwidth (float, optional): The bandwidth parameter for the Gaussian kernel. Defaults to 1.0.

    Returns:
        None

    This function estimates the density of a set of points using a 2D Gaussian kernel density estimation. It uses the
    `gaussian_kde` function from the `scipy.stats` module to perform the estimation. The resulting density is
    visualized using a heatmap plot. The input points are plotted as red dots on the plot.
    """

    kde = gaussian_kde(points.T, bw_method=bandwidth)
    x_min, y_min = points.min(axis=0) - 10
    x_max, y_max = points.max(axis=0) + 10
    x, y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([x.ravel(), y.ravel()])
    density = np.reshape(kde(positions).T, x.shape)
    
    plt.imshow(np.rot90(density), cmap=plt.cm.gist_earth_r, extent=[x_min, x_max, y_min, y_max])
    plt.scatter(points[:, 0], points[:, 1], c='red', s=2)
    plt.title(f"2D Gaussian Kernel Density Estimation for {method_name}")
    plt.colorbar()
    plt.show()


def calculate_completeness_and_correctness(ground_truth_image, gray_scale_image, filtered_points):
    """
    Calculates the completeness and correctness of a ground truth image based on the filtered points.

    Parameters:
        ground_truth_image (numpy.ndarray): The ground truth image.
        gray_scale_image (numpy.ndarray): The gray scale image.
        filtered_points (numpy.ndarray): The filtered points.

    Returns:
        tuple: A tuple containing the completeness and correctness values.

    This function calculates the completeness and correctness of a ground truth image based on the filtered points.
    It first converts the gray scale image to gray scale and identifies the ground truth points. It then plots the
    ground truth points on an image using a scatter plot. The completeness is calculated by comparing the filtered
    points with the ground truth points. The correctness is calculated by comparing the filtered points with the
    ground truth points and counting the true positives (TP), false positives (FP), and false negatives (FN).
    The completeness is calculated as TP / (TP + FN) and the correctness is calculated as TP / (TP + FP).
    """

    gray_scale_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)
    mask = gray_scale_image > 0

    ground_truth_points = np.argwhere(mask)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(gray_scale_image, cv2.COLOR_GRAY2RGB))
    plt.scatter(ground_truth_points[:, 1], ground_truth_points[:, 0], c='blue', s=2)
    plt.title("Ground truth points")
    plt.show()

    TP = FP = FN = 0

    # Convert filtered_points to a set for faster lookup
    ground_truth_points_set = set(map(tuple, ground_truth_points.tolist()))
    filtered_points_set = set(map(tuple, filtered_points.tolist()))

    for gt_point in ground_truth_points_set:
        if tuple(gt_point) in filtered_points_set:
            TP += 1
        else:
            FN += 1

    for fp in filtered_points_set:
        if tuple(fp) not in ground_truth_points_set:
            FP += 1

    return TP / (TP + FN), TP / (TP + FP)


image_path = './DLR_AerialCrowdDataset/Train/Images/I_2.jpg'
ground_truth_image_path = './DLR_AerialCrowdDataset/Train/Annotation/I_2.png'
image = cv2.imread(image_path)
ground_truth_image = cv2.imread(ground_truth_image_path)

# resize the gray_scale_image and ground truth gray_scale_image to 720 X 480 pixels
image = cv2.resize(image, (720, 480), interpolation=cv2.INTER_AREA)
ground_truth_image = cv2.resize(ground_truth_image, (720, 480), interpolation=cv2.INTER_AREA)

# display the original image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.show()

# convert to gray scale to apply Fast 9 method
gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# get the corners using fast 9 method
original_corners = original_fast(gray_scale_image, threshold=30)

# filter the points to keep only the points that belong to the crowd
original_filtered_points = crowd_features_filteration(gray_scale_image, original_corners, radius=100, num_sectors=20, min_points_per_sector=50, no_of_clusters=5, method_name="FAST 9 method")

# calculate the completness and correctness using the ground truth gray_scale_image. Compare the filtered points with the ground truth points
completeness, correctness = calculate_completeness_and_correctness(ground_truth_image, gray_scale_image, original_filtered_points)

# print accuracy of FAST 9 method
print("original method:", completeness, correctness)

# visualize the density estimation for the original FAST 9 method
estimate_density(original_filtered_points, method_name="FAST 9 method")

# get the corners using the base paper method
custom_corners = custom_fast(gray_scale_image, threshold=30)

# filter the points to keep only the points that belong to the crowd
custom_filtered_points = crowd_features_filteration(gray_scale_image, custom_corners, radius=100, num_sectors=20, min_points_per_sector=50, no_of_clusters=5, method_name="Base paper method")

# calculate the completness and correctness using the ground truth gray_scale_image. Compare the filtered points with the ground truth points
completeness, correctness = calculate_completeness_and_correctness(ground_truth_image, gray_scale_image, custom_filtered_points)

# print accuracy of base paper method
print("custom method:", completeness, correctness)

# visualize the density estimation for the base paper method
estimate_density(custom_filtered_points, method_name="Base paper method")





