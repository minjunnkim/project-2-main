"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    centroid = np.mean(points, axis=0)  # Compute centroid (mean x and y)
    
    std_x = np.std(points[:, 0])  # Compute standard deviation for x
    std_y = np.std(points[:, 1])  # Compute standard deviation for y

    scale_x = 1.0 / std_x  # Scale factor for x
    scale_y = 1.0 / std_y  # Scale factor for y
    
    # Construct transformation matrix
    T = np.array([
        [scale_x, 0, -scale_x * centroid[0]],
        [0, scale_y, -scale_y * centroid[1]],
        [0, 0, 1]
    ])
    
    # Convert points to homogeneous coordinates
    points_homog = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Apply transformation matrix
    points_normalized_homog = (T @ points_homog.T).T  # Apply transformation
    
    # Convert back to non-homogeneous form
    points_normalized = points_normalized_homog[:, :2]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    F_orig = T_b.T @ F_norm @ T_a

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def make_singular(F_norm: np.array) -> np.ndarray:
    """
    Force F to be singular by zeroing the smallest of its singular values.
    This is done because F is not supposed to be full rank, but an inaccurate
    solution may end up as rank 3.

    Args:
    - F_norm: A numpy array of shape (3,3) representing the normalized fundamental matrix.

    Returns:
    - F_norm_s: A numpy array of shape (3, 3) representing the normalized fundamental matrix
                with only rank 2.
    """
    U, D, Vt = np.linalg.svd(F_norm)
    D[-1] = 0
    F_norm_s = np.dot(np.dot(U, np.diag(D)), Vt)

    return F_norm_s


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here. Equation (9) in the documentation indicates
    one equation of a linear system in which you'll want to solve for f_{i, j}.

    Since the matrix is defined up to a scale, many solutions exist. To constrain
    your solution, use can either use SVD and use the last Vt vector as your
    solution, or you can fix f_{3, 3} to be 1 and solve with least squares.

    Be sure to reduce the rank of your estimate - it should be rank 2. The
    make_singular() function can do this for you.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # Normalize the points
    points_a_normalized, T_a = normalize_points(points_a)
    points_b_normalized, T_b = normalize_points(points_b)

    # Construct the equation system Ax = 0
    A = np.zeros((points_a.shape[0], 9))
    for i in range(points_a.shape[0]):
        u, v = points_a_normalized[i]
        u_prime, v_prime = points_b_normalized[i]
        A[i] = [u_prime * u, u_prime * v, u_prime, v_prime * u, v_prime * v, v_prime, u, v, 1]

    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    F_normalized = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    F_normalized = make_singular(F_normalized)

    # Unnormalize the fundamental matrix
    F = unnormalize_F(F_normalized, T_a, T_b)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
