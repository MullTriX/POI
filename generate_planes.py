import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from csv import writer

def generate_plane(length: int, width: int, orientation: str) -> tuple:
    """ Generates a plane in 3D space with the given length and width.

    Args:
        length (int): Length of the plane.
        width (int): Width of the plane.
        orientation (str): Orientation of the plane. Can be 'xy', 'yz', or 'xz'.

    Returns:
        tuple: Three lists of x, y, and z coordinates of the points on the plane.
    """
    num_points = 2000
    
    if orientation == 'xy':
        x_range = (-length, length)
        y_range = (-width, width)
        z_range = (0, 0)
    elif orientation == 'yz':
        x_range = (0, 0)
        y_range = (-length, length)
        z_range = (-width, width)
    elif orientation == 'xz':
        x_range = (-length, length)
        y_range = (0, 0)
        z_range = (-width, width)
    else:
        raise ValueError("Invalid orientation. Choose from 'xy', 'yz', or 'xz'.")
    
    x = np.random.uniform(x_range[0], x_range[1], num_points)
    y = np.random.uniform(y_range[0], y_range[1], num_points)
    z = np.random.uniform(z_range[0], z_range[1], num_points)
    return (x, y, z)

def generate_cylinder(radius: float, height: float) -> tuple:
    """ Generates a cylinder in 3D space with the given radius and height.

    Args:
        radius (float): Radius of the cylinder.
        height (float): Height of the cylinder.

    Returns:
        tuple: Three lists of x, y, and z coordinates of the points on the cylinder.
    """
    num_points = 5000
    
    theta = np.linspace(0, 2 * np.pi, num_points)
    z = np.random.uniform(0, height, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return (x, y, z)

def save_points_to_file(filename: str, points: zip) -> None:
    """ Saves the points to a file in XYZ format.

    Args:
        filename (str): Name of the file to save the points.
        points (zip): Tuple containing three lists of x, y, and z coordinates.
    """
    with open(filename, 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)
        for point in points:
            csvwriter.writerow(point)

# Generate and save the xy plane
x, y, z = generate_plane(10, 10, orientation='xy')
points = zip(x, y, z)
save_points_to_file('Data/plane_xy.xyz', points)

# Generate and save the yz plane
x, y, z = generate_plane(10, 10, orientation='yz')
points = zip(x, y, z)
save_points_to_file('Data/plane_yz.xyz', points)

# Generate and save the cylinder
x, y, z = generate_cylinder(5, 20)
points = zip(x, y, z)
save_points_to_file('Data/cylinder.xyz', points)
