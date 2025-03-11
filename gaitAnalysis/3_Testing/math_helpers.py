import numpy as np

def gaussian_2d(x, y, sigma, width, height):
    """
    Create a 2D Gaussian distribution centered at (x, y).

    :param x: The x-coordinate of the center.
    :param y: The y-coordinate of the center.
    :param sigma: The standard deviation of the Gaussian.
    :param width: The width of the 2D array.
    :param height: The height of the 2D array.
    :return: A 2D Gaussian distribution.
    """
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return gaussian

def multiply_with_gaussian(array, x, y, sigma):
    """
    Multiply a 2D array with a Gaussian centered at (x, y).

    :param array: The 2D array to be multiplied.
    :param x: The x-coordinate of the Gaussian center.
    :param y: The y-coordinate of the Gaussian center.
    :param sigma: The standard deviation of the Gaussian.
    :return: The resulting 2D array after multiplication.
    """
    height, width = array.shape
    gaussian = gaussian_2d(x, y, sigma, width, height)
    return array * gaussian

def gaussian_brightness(frame, x, y, half_size=7):
    
    """
    Calculate the brightness of a square of pixels centered at (x, y) in a frame.
    
    :param frame: The frame containing the square of pixels.
    :param x: The x-coordinate of the center of the square.
    :param y: The y-coordinate of the center of the square.
    :param half_size: The half-size of the square.
    
    :return: The square of pixels and the brightness of the square.
    """
    
    # Ensure the coordinates are within the image bounds
    height, width, _ = frame.shape
    x_min = max(x - half_size, 0)
    x_max = min(x + half_size, width)
    y_min = max(y - half_size, 0)
    y_max = min(y + half_size, height)

    # Capture the square of pixels
    square = frame[y_min:y_max, x_min:x_max,0]

    frame = multiply_with_gaussian(square, x - x_min, y - y_min, sigma=half_size/2)
    brightness = np.average(frame.flatten())
    return brightness

def max_brightness(frame, x, y, half_size=7):
    """
    Calculate the maximum brightness of a square of pixels centered at (x, y) in a frame.
    
    :param frame: The frame containing the square of pixels.
    :param x: The x-coordinate of the center of the square.
    :param y: The y-coordinate of the center of the square.
    :param half_size: The half-size of the square.
    
    :return: The maximum brightness of the square.
    """
    # Ensure the coordinates are within the image bounds
    height, width, _ = frame.shape
    x_min = max(x - half_size, 0)
    x_max = min(x + half_size, width)
    y_min = max(y - half_size, 0)
    y_max = min(y + half_size, height)

    # Capture the square of pixels
    square = frame[y_min:y_max, x_min:x_max,0]

    brightness = np.max(square)
    return brightness

def find_midpoint(x1, y1, x2, y2):
    """
    Find the midpoint of a line defined by two points (x1, y1) and (x2, y2).
    
    :param x1: The x-coordinate of the first point.
    :param y1: The y-coordinate of the first point.
    :param x2: The x-coordinate of the second point.
    :param y2: The y-coordinate of the second point.
    
    :return: The midpoint of the line.
    """
    return (x1 + x2) / 2, (y1 + y2) / 2

def find_perpendicular_slope(x1, y1, x2, y2):
    """
    Find the slope of the perpendicular bisector of a line defined by two points (x1, y1) and (x2, y2).
    
    :param x1: The x-coordinate of the first point.
    :param y1: The y-coordinate of the first point.
    :param x2: The x-coordinate of the second point.
    :param y2: The y-coordinate of the second point.
    
    :return: The slope of the perpendicular bisector.
    """
    if x2 == x1:  # Vertical line
        return 0  # Perpendicular bisector is horizontal
    elif y2 == y1:  # Horizontal line
        return float('inf')  # Perpendicular bisector is vertical
    else:
        slope = (y2 - y1) / (x2 - x1)
        return -1 / slope
    
def find_perpendicular_bisector(frame_height, line):
    """
    Find the perpendicular bisector of a line defined by two points (x1, y1) and (x2, y2).

    :param frame_height: The height of the frame.
    :param line: The line defined by two points.

    :return: The perpendicular bisector of the line.
    
    """
    x1, y1, x2, y2 = line
    
    mid_x, mid_y = find_midpoint(x1, y1, x2, y2)
    perp_slope = find_perpendicular_slope(x1, y1, x2, y2)
    
    if perp_slope == float('inf'):
        # vertical line
        return (int(mid_x), 0, int(mid_x), frame_height)
    else:
        # Calculate two points on the perpendicular bisector
        length = 1000  # Length of the line to draw
        dx = length / np.sqrt(1 + perp_slope**2)
        dy = perp_slope * dx
        return int(mid_x - dx), int(mid_y - dy), int(mid_x + dx), int(mid_y + dy)

def find_intersection(line1, line2, tolerance=10):
    """
    Find the intersection point of two lines.
    Each line is defined by two points (x1, y1) and (x2, y2).
    
    :param line1: The first line defined by two points.
    :param line2: The second line defined by two points.
    
    :return: The intersection point of the two lines.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate the denominators
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < tolerance:
        return None  # Lines are parallel or coincident

    # Calculate the intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    return (int(px), int(py))

def find_angle(line1, line2):
    """
    Find the angle between two lines.
    Each line is defined by two points (x1, y1) and (x2, y2).
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Calculate the slopes of the lines
    if x2 - x1 == 0:  # Vertical line
        slope1 = float('inf')
    else:
        slope1 = (y2 - y1) / (x2 - x1)
    
    if x4 - x3 == 0:  # Vertical line
        slope2 = float('inf')
    else:
        slope2 = (y4 - y3) / (x4 - x3)

    print('slope1, slope2:')
    print(slope1, slope2)
    
    # Handle the case where both lines are vertical
    if slope1 == float('inf') and slope2 == float('inf'):
        return 0.0
    
    # Calculate the angle between the lines
    if 1 + slope1 * slope2 == 0:
        return 90.0
    
    # Calculate the angle between the lines
    if slope1 == float('inf'):
        angle = np.arctan(slope2)
    elif slope2 == float('inf'):
        angle = np.arctan(slope1)
    else:
        angle = np.arctan((slope2 - slope1) / (1 + slope1 * slope2))
    
    return np.degrees(angle)

def calculate_angle(pointA, pointB, pointC):
    """
    Calculate the angle ABC (in degrees) formed by three points A, B, and C.
    The angle is measured in the range [0, 360) degrees.
    
    Parameters:
    pointA, pointB, pointC: Tuples representing the coordinates of the points (x, y)
    
    Returns:
    angle: The angle ABC in degrees
    """
    # Convert points to numpy arrays
    A = np.array(pointA)
    B = np.array(pointB)
    C = np.array(pointC)
    
    # Calculate vectors BA and BC
    BA = A - B
    BC = C - B
    
    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    
    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_BA * magnitude_BC)
    
    # Ensure the cosine value is within the valid range [-1, 1] to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle_radians = np.arccos(cos_angle)
    
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    # Determine the direction of the angle using the cross product
    cross_product = np.cross(BA, BC)
    if cross_product < 0:
        # angle_degrees = 360 - angle_degrees
        # opposite angle will be negative
        angle_degrees = -angle_degrees
    
    return angle_degrees

def rotate_to_align_with_y_axis(a, b): # use head and desired vector
    """
    Rotates 2D vector `a` such that 2D vector `b` aligns with the y-axis.
    
    Parameters:
    - a: Vector to rotate (numpy array of shape (2,))
    - b: Vector to align with y-axis (numpy array of shape (2,))
    
    Returns:
    - Rotated vector a (numpy array of shape (2,))
    """
    # Normalize vector b
    b = b / np.linalg.norm(b)
    
    # Compute the angle to rotate b to align with the y-axis
    angle_to_y = np.arctan2(b[0], b[1])  # Angle between b and the y-axis
    
    # Construct the 2D rotation matrix to align b with the y-axis
    R = np.array([
        [np.cos(angle_to_y), -np.sin(angle_to_y)],
        [np.sin(angle_to_y), np.cos(angle_to_y)]
    ])
    
    # Rotate vector a using the rotation matrix
    a_rotated = np.dot(R, a)
    return a_rotated

def get_rot_angle_y_axis(b): # b is the desired vector
    # Normalize vector b
    b = b / np.linalg.norm(b)
    
    # Compute the angle to rotate b to align with the y-axis
    angle_to_y = np.arctan2(b[0], b[1])  # Angle between b and the y-axis

    return -angle_to_y
