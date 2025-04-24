import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from csv import reader
import os
import pyransac3d as pyrsc

def load_xyz_file(filepath : str) -> np.ndarray:
    """
    Loads a .xyz file containing 3D points.

    Args:
        filepath (str): Path to the .xyz file.

    Returns:
        np.ndarray: array of 3D points.
    """
    with open(filepath, 'r') as file:
        csv_reader = reader(file, delimiter=',')
        points = np.array([list(map(float, row)) for row in csv_reader])
    return points

def ransac_plane_fitting(points: np.ndarray, threshold: float = 0.01, max_iterations: int = 1000) -> tuple:
    """
    Fits a plane to a point cloud using the RANSAC algorithm.

    Args:
        points (np.ndarray): array of 3D points.
        threshold (float): Distance threshold to consider a point as an inlier.
        max_iterations (int): Maximum number of iterations.

    Returns:
        tuple: Best plane parameters (a, b, c, d) and the inliers.
    """
    best_plane = None
    best_inliers = None
    max_inliers = 0

    for _ in range(max_iterations):
        sample = points[np.random.choice(points.shape[0], 3, replace=False)]

        # Compute the plane equation
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) == 0:
            continue

        a, b, c = normal
        d = -np.dot(normal, sample[0])

        # Compute distances of all points to the plane
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.linalg.norm(normal)

        # Identify inliers
        inliers = distances < threshold
        num_inliers = np.sum(inliers)

        # Update the best plane
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_plane = (a, b, c, d)
            best_inliers = inliers

    return best_plane, best_inliers

def package_ransac_plane_fitting(points: np.ndarray, threshold: float = 0.01, max_iterations: int = 1000) -> tuple:
    plane = pyrsc.Plane()
    best_eq, best_inliers = plane.fit(points, threshold, max_iterations)
    return best_eq, best_inliers

def classify_plane(plane: tuple, points: np.ndarray, inliers: np.ndarray, threshold: float = 0.01) -> str:
    """
    Classifies a plane as horizontal, vertical, or not a plane.

    Args:
        plane (tuple): Plane parameters (a, b, c, d).
        points (np.ndarray): array of 3D points.
        inliers (np.ndarray): Boolean mask of inliers.
        threshold (float): Threshold for average distance to classify as a plane.

    Returns:
        str: Classification result ("horizontal", "vertical", or "not a plane").
    """
    a, b, c, _ = plane
    normal = np.array([a, b, c])
    avg_distance = np.mean(np.abs(a * points[inliers, 0] + b * points[inliers, 1] + c * points[inliers, 2]) / np.linalg.norm(normal))

    if avg_distance > threshold:
        return "not a plane"

    # Check orientation
    if np.isclose(c, 0, atol=1e-2):
        return "vertical"
    else:
        return "horizontal"

def process_each_cluster(points: np.ndarray, labels: np.ndarray, threshold: float = 0.01, max_iterations: int = 1000) -> None:
    """
    Processes each cluster of points to fit a plane and classify it.

    Args:
        points (np.ndarray): array of 3D points.
        cluster_id (int): ID of the cluster.
        threshold (float): Distance threshold for RANSAC.
        max_iterations (int): Maximum number of iterations for RANSAC.
    """
    for cluster_id in range(3):
            cluster_points = points[labels == cluster_id]
            print(f"Processing cluster {cluster_id + 1}")
            print("----> SELF-IMPLEMENTED RANSAC <----")
            plane, inliers = ransac_plane_fitting(cluster_points, threshold, max_iterations)
            classification = classify_plane(plane, cluster_points, inliers)
            
            print(f"  Plane parameters (a, b, c, d): {plane}")
            print(f"  Number of inliers: {np.sum(inliers)}")
            print(f"  Classification: {classification}")
            
            print("----> PACKAGE RANSAC <----")
            plane, inliers = package_ransac_plane_fitting(cluster_points, threshold, max_iterations)
            classification = classify_plane(plane, cluster_points, inliers)
            print(f"  Plane parameters (a, b, c, d): {plane}")
            print(f"  Number of inliers: {np.sum(inliers)}")
            print(f"  Classification: {classification}\n")



filepath = "Data/"
for files in os.listdir(filepath):
    if files.endswith(".xyz"):
        fullpath = os.path.join(filepath, files)
        print("=======================================================================")
        print(f"Processing file: {fullpath}")
        print("=======================================================================")
        
        points = load_xyz_file(fullpath)
        
        # Cluster the points using K-Means
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(points)
        
        # Process each cluster
        process_each_cluster(points=points, labels=labels, threshold=0.01, max_iterations=1000)