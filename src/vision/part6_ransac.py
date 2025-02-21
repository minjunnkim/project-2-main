import math

import numpy as np
from vision.part5_fundamental_matrix import estimate_fundamental_matrix


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: float
) -> int:
    """
    Calculates the number of RANSAC iterations needed for a given guarantee of
    success.

    Args:
        prob_success: float representing the desired guarantee of success
        sample_size: int the number of samples included in each RANSAC
            iteration
        ind_prob_success: float representing the probability that each element
            in a sample is correct

    Returns:
        num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    prob_all_inliers = ind_prob_correct ** sample_size

    num_samples = math.log(1 - prob_success) / math.log(1 - prob_all_inliers)

    num_samples = math.ceil(num_samples)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)


def ransac_fundamental_matrix(
    matches_a: np.ndarray, matches_b: np.ndarray
) -> np.ndarray:
    """
    For this section, use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You would reuse
    estimate_fundamental_matrix() from part 2 of this assignment and
    calculate_num_ransac_iterations().

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 30 points for either left or
    right images.

    Tips:
        0. You will need to determine your prob_success, sample_size, and
            ind_prob_success values. What is an acceptable rate of success? How
            many points do you want to sample? What is your estimate of the
            correspondence accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for creating
            your random samples.
        2. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 0.1.

    Args:
        matches_a: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image A
        matches_b: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
        best_F: A numpy array of shape (3, 3) representing the best fundamental
            matrix estimation
        inliers_a: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image A that are inliers with respect to
            best_F
        inliers_b: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image B that are inliers with respect to
            best_F
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    num_iterations = calculate_num_ransac_iterations(prob_success=0.99, sample_size=8, ind_prob_correct=0.5)
    best_F = None
    max_inliers = 0
    inliers_a = None
    inliers_b = None
    threshold = 0.1

    for _ in range(num_iterations):
        # random sample 8 correspondences
        sample_indices = np.random.choice(matches_a.shape[0], size=8, replace=False)
        sampled_a = matches_a[sample_indices]
        sampled_b = matches_b[sample_indices]

        # estimate fundamental matrix
        F = estimate_fundamental_matrix(sampled_a, sampled_b)

        # compute error
        ones = np.ones((matches_a.shape[0], 1))
        homogeneous_a = np.hstack((matches_a, ones))
        homogeneous_b = np.hstack((matches_b, ones))

        # compute epipolar constraint errors
        errors = np.abs(np.sum(homogeneous_b * (F @ homogeneous_a.T).T, axis=1))

        # identify inliers
        inlier_mask = errors < threshold
        num_inliers = np.sum(inlier_mask)

        # update best model
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_F = F
            inliers_a = matches_a[inlier_mask]
            inliers_b = matches_b[inlier_mask]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_F, inliers_a, inliers_b
