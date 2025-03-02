#!/usr/bin/python3

import copy
import pickle
from typing import Any, Callable, List, Optional, Tuple, Union


import numpy as np
import PIL
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from IPython.core.debugger import set_trace
from matplotlib.axes._axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from vision.part4_projection_matrix import projection


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Bring image values to [0,1] range

    Args:
        img: (H,W,C) or (H,W) image
    """
    img -= img.min()
    img /= img.max()
    return img


def verify(function: Callable) -> str:
    """Will indicate with a print statement whether assertions passed or failed
    within function argument call.
    Args:
        function: Python function object
    Returns:
        string that is colored red or green when printed, indicating success
    """
    try:
        function()
        return '\x1b[32m"Correct"\x1b[0m'
    except AssertionError:
        return '\x1b[31m"Wrong"\x1b[0m'


def rgb2gray(img: np.ndarray) -> np.ndarray:
    """Use the coefficients used in OpenCV, found here:
    https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    Args:
        Numpy array of shape (M,N,3) representing RGB image in HWC format

    Returns:
        Numpy array of shape (M,N) representing grayscale image
    """
    # Grayscale coefficients
    c = [0.299, 0.587, 0.114]
    return img[:, :, 0] * c[0] + img[:, :, 1] * c[1] + img[:, :, 2] * c[2]


def PIL_resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        img: Array representing an image
        size: Tuple representing new desired (width, height)

    Returns:
        img
    """
    img = numpy_arr_to_PIL_image(img, scale_to_255=True)
    img = img.resize(size)
    img = PIL_image_to_numpy_arr(img)
    return img


def PIL_image_to_numpy_arr(img: Image, downscale_by_255: bool = True) -> np.ndarray:
    """
    Args:
        img: PIL Image
        downscale_by_255: whether to divide uint8 values by 255 to normalize
        values to range [0,1]

    Returns:
        img
    """
    img = np.asarray(img)
    img = img.astype(np.float32)
    if downscale_by_255:
        img /= 255
    return img


def im2single(im: np.ndarray) -> np.ndarray:
    """
    Args:
        img: uint8 array of shape (m,n,c) or (m,n) and in range [0,255]

    Returns:
        im: float or double array of identical shape and in range [0,1]
    """
    im = im.astype(np.float32) / 255
    return im


def single2im(im: np.ndarray) -> np.ndarray:
    """
    Args:
        im: float or double array of shape (m,n,c) or (m,n) and in range [0,1]

    Returns:
        im: uint8 array of identical shape and in range [0,255]
    """
    im *= 255
    im = im.astype(np.uint8)
    return im


def numpy_arr_to_PIL_image(img: np.ndarray, scale_to_255: False) -> PIL.Image:
    """
    Args:
        img: in [0,1]

    Returns:
        img in [0,255]
    """
    if scale_to_255:
        img *= 255
    return PIL.Image.fromarray(np.uint8(img))


def load_image(path: str) -> np.ndarray:
    """
    Args:
        path: string representing a file path to an image

    Returns:
        float_img_rgb: float or double array of shape (m,n,c) or (m,n)
           and in range [0,1], representing an RGB image
    """
    img = PIL.Image.open(path)
    img = np.asarray(img, dtype=float)
    float_img_rgb = im2single(img)
    return float_img_rgb


def save_image(path: str, im: np.ndarray) -> None:
    """
    Args:
        path: string representing a file path to an image
        img: numpy array
    """
    folder_path = os.path.split(path)[0]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    img = copy.deepcopy(im)
    img = single2im(img)
    pil_img = numpy_arr_to_PIL_image(img, scale_to_255=False)
    pil_img.save(path)


def cheat_interest_points(eval_file, scale_factor):
    """
    This function is provided for development and debugging but cannot be used
    in the final hand-in. It 'cheats' by generating interest points from known
    correspondences. It will only work for the 3 image pairs with known
    correspondences.

    Args:
    - eval_file: string representing the file path to the list of known
      correspondences
    - scale_factor: Python float representing the scale needed to map from the
      original image coordinates to the resolution being used for the current
      experiment.

    Returns:
    - x1: A numpy array of shape (k,) containing ground truth x-coordinates of
      imgA correspondence pts
    - y1: A numpy array of shape (k,) containing ground truth y-coordinates of
      imgA correspondence pts
    - x2: A numpy array of shape (k,) containing ground truth x-coordinates of
      imgB correspondence pts
    - y2: A numpy array of shape (k,) containing ground truth y-coordinates of
      imgB correspondence pts
    """
    with open(eval_file, "rb") as f:
        d = pickle.load(f, encoding="latin1")

    return d["x1"] * scale_factor, d["y1"] * scale_factor, d["x2"] * scale_factor, d["y2"] * scale_factor


def hstack_images(img1, img2):
    """
    Stacks 2 images side-by-side and creates one combined image.

    Args:
    - imgA: A numpy array of shape (M,N,3) representing rgb image
    - imgB: A numpy array of shape (D,E,3) representing rgb image

    Returns:
    - newImg: A numpy array of shape (max(M,D), N+E, 3)
    """

    # CHANGED
    imgA = np.array(img1)
    imgB = np.array(img2)
    Height = max(imgA.shape[0], imgB.shape[0])
    Width = imgA.shape[1] + imgB.shape[1]

    newImg = np.zeros((Height, Width, 3), dtype=imgA.dtype)
    newImg[: imgA.shape[0], : imgA.shape[1], :] = imgA
    newImg[: imgB.shape[0], imgA.shape[1] :, :] = imgB

    # newImg = PIL.Image.fromarray(np.uint8(newImg))
    return newImg


def show_interest_points(img: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Visualized interest points on an image with random colors

    Args:
        img: array of shape (M,N,C)
        X: array of shape (k,) containing x-locations of interest points
        Y: array of shape (k,) containing y-locations of interest points

    Returns:
        newImg: A numpy array of shape (M,N,C) showing the original image with
            colored circles at keypoints plotted on top of it
    """
    # CHANGED
    newImg = img.copy()
    newImg = numpy_arr_to_PIL_image(newImg, True)
    r = 10
    draw = PIL.ImageDraw.Draw(newImg)
    for x, y in zip(X.astype(int), Y.astype(int)):
        cur_color = np.random.rand(3) * 255
        cur_color = (int(cur_color[0]), int(cur_color[1]), int(cur_color[2]))
        # newImg = cv2.circle(newImg, (x, y), 10, cur_color, -1, cv2.LINE_AA
        draw.ellipse([x - r, y - r, x + r, y + r], fill=cur_color)

    return PIL_image_to_numpy_arr(newImg, True)


def show_correspondence_circles(imgA, imgB, X1, Y1, X2, Y2):
    """
    Visualizes corresponding points between two images by plotting circles at
    each correspondence location. Corresponding points will have the same
    random color.

    Args:
        imgA: A numpy array of shape (M,N,3)
        imgB: A numpy array of shape (D,E,3)
        x1: A numpy array of shape (k,) containing x-locations of imgA keypoints
        y1: A numpy array of shape (k,) containing y-locations of imgA keypoints
        x2: A numpy array of shape (j,) containing x-locations of imgB keypoints
        y2: A numpy array of shape (j,) containing y-locations of imgB keypoints

    Returns:
        newImg: A numpy array of shape (max(M,D), N+E, 3)
    """
    # CHANGED
    newImg = hstack_images(imgA, imgB)
    newImg = numpy_arr_to_PIL_image(newImg, True)
    draw = PIL.ImageDraw.Draw(newImg)
    shiftX = imgA.shape[1]
    X1 = X1.astype(int)
    Y1 = Y1.astype(int)
    X2 = X2.astype(int)
    Y2 = Y2.astype(int)
    r = 10
    for x1, y1, x2, y2 in zip(X1, Y1, X2, Y2):
        cur_color = np.random.rand(3) * 255
        cur_color = (int(cur_color[0]), int(cur_color[1]), int(cur_color[2]))
        green = (0, 1, 0)
        draw.ellipse([x1 - r + 1, y1 - r + 1, x1 + r - 1, y1 + r - 1], fill=cur_color, outline=green)
        draw.ellipse([x2 + shiftX - r + 1, y2 - r + 1, x2 + shiftX + r - 1, y2 + r - 1], fill=cur_color, outline=green)

        # newImg = cv2.circle(newImg, (x1, y1), 10, cur_color, -1, cv2.LINE_AA)
        # newImg = cv2.circle(newImg, (x1, y1), 10, green, 2, cv2.LINE_AA)
        # newImg = cv2.circle(newImg, (x2+shiftX, y2), 10, cur_color, -1,
        #                     cv2.LINE_AA)
        # newImg = cv2.circle(newImg, (x2+shiftX, y2), 10, green, 2, cv2.LINE_AA)

    return PIL_image_to_numpy_arr(newImg, True)


def show_correspondence_lines(imgA, imgB, X1, Y1, X2, Y2, line_colors=None):
    """
    Visualizes corresponding points between two images by drawing a line
    segment between the two images for each (x1,y1) (x2,y2) pair.

    Args:
        imgA: A numpy array of shape (M,N,3)
        imgB: A numpy array of shape (D,E,3)
        x1: A numpy array of shape (k,) containing x-locations of imgA keypoints
        y1: A numpy array of shape (k,) containing y-locations of imgA keypoints
        x2: A numpy array of shape (j,) containing x-locations of imgB keypoints
        y2: A numpy array of shape (j,) containing y-locations of imgB keypoints
        line_colors: A numpy array of shape (N x 3) with colors of correspondence
            lines (optional)

    Returns:
        newImg: A numpy array of shape (max(M,D), N+E, 3)
    """
    newImg = hstack_images(imgA, imgB)
    newImg = numpy_arr_to_PIL_image(newImg, True)

    draw = PIL.ImageDraw.Draw(newImg)
    r = 10
    shiftX = imgA.shape[1]
    X1 = X1.astype(int)
    Y1 = Y1.astype(int)
    X2 = X2.astype(int)
    Y2 = Y2.astype(int)

    dot_colors = (np.random.rand(len(X1), 3) * 255).astype(int)
    if line_colors is None:
        line_colors = dot_colors
    else:
        line_colors = (line_colors * 255).astype(int)

    for x1, y1, x2, y2, dot_color, line_color in zip(X1, Y1, X2, Y2, dot_colors, line_colors):
        # newImg = cv2.circle(newImg, (x1, y1), 5, dot_color, -1)
        # newImg = cv2.circle(newImg, (x2+shiftX, y2), 5, dot_color, -1)
        # newImg = cv2.line(newImg, (x1, y1), (x2+shiftX, y2), line_color, 2,
        #                                     cv2.LINE_AA)
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=tuple(dot_color))
        draw.ellipse((x2 + shiftX - r, y2 - r, x2 + shiftX + r, y2 + r), fill=tuple(dot_color))
        draw.line((x1, y1, x2 + shiftX, y2), fill=tuple(line_color), width=10)
    return PIL_image_to_numpy_arr(newImg, True)


def show_ground_truth_corr(imgA: str, imgB: str, corr_file: str, show_lines: bool = True):
    """
    Show the ground truth correspondeces

    Args:
        imgA: string, representing the filepath to the first image
        imgB: string, representing the filepath to the second image
        corr_file: filepath to pickle (.pkl) file containing the correspondences
        show_lines: boolean, whether to visualize the correspondences as line segments
    """
    imgA = load_image(imgA)
    imgB = load_image(imgB)
    with open(corr_file, "rb") as f:
        d = pickle.load(f)
    if show_lines:
        return show_correspondence_lines(imgA, imgB, d["x1"], d["y1"], d["x2"], d["y2"])
    else:
        # show circles
        return show_correspondence_circles(imgA, imgB, d["x1"], d["y1"], d["x2"], d["y2"])


def load_corr_pkl_file(corr_fpath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Load ground truth correspondences from a pickle (.pkl) file. """
    original = corr_fpath
    destination = corr_fpath[: -4] + '_unix.pkl'

    with open(original, 'rb') as infile:
        content = infile.read()
    with open(destination, 'wb') as output:
        for line in content.splitlines():
            output.write(line + str.encode('\n'))
    with open(destination, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    x1 = d["x1"].squeeze()
    y1 = d["y1"].squeeze()
    x2 = d["x2"].squeeze()
    y2 = d["y2"].squeeze()

    return x1, y1, x2, y2


def evaluate_correspondence(
    imgA: np.ndarray,
    imgB: np.ndarray,
    corr_fpath: str,
    scale_factor: float,
    x1_est: np.ndarray,
    y1_est: np.ndarray,
    x2_est: np.ndarray,
    y2_est: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    num_req_matches: int = 100,
) -> Tuple[float, np.ndarray]:
    """
    Function to evaluate estimated correspondences against ground truth.

    The evaluation requires 100 matches to receive full credit
    when num_req_matches=100 because we define accuracy as:

    Let TP = true_pos
    Let FP = false_pos

    Accuracy = (TP)/(TP + FP) * min(num_matches,num_req_matches)/num_req_matches

    Args:
        imgA: A numpy array of shape (M,N,C) representing a first image
        imgB: A numpy array of shape (M,N,C) representing a second image
        corr_fpath: string, representing a filepath to a .pkl file containing
            ground truth correspondences
        scale_factor: scale factor on the size of the images
        x1_est: array of shape (k,) containing estimated x-coordinates of imgA correspondence pts
        y1_est: array of shape (k,) containing estimated y-coordinates of imgA correspondence pts
        x2_est: array of shape (k,) containing estimated x-coordinates of imgB correspondence pts
        y2_est: array of shape (k,) containing estimated y-coordinates of imgB correspondence pts
        confidences: (optional) confidence values in the matches

    Returns:
        acc: accuracy as decimal / ratio (between 0 and 1)
        rendered_img: image with correct matches rendered as green lines, incorrect rendered as red
    """
    if confidences is None:
        confidences = np.random.rand(len(x1_est))
        confidences /= np.max(confidences)

    x1_est = x1_est.squeeze() / scale_factor
    y1_est = y1_est.squeeze() / scale_factor
    x2_est = x2_est.squeeze() / scale_factor
    y2_est = y2_est.squeeze() / scale_factor

    num_matches = x1_est.shape[0]

    x1, y1, x2, y2 = load_corr_pkl_file(corr_fpath)

    good_matches = [False for _ in range(len(x1_est))]
    # array marking which GT pairs are already matched
    matched = [False for _ in range(len(x1))]

    # iterate through estimated pairs in decreasing order of confidence
    priority = np.argsort(-confidences)
    for i in priority:
        # print('Examining ({:4.0f}, {:4.0f}) to ({:4.0f}, {:4.0f})'.format(
        #     x1_est[i], y1_est[i], x2_est[i], y2_est[i]))
        cur_offset = np.asarray([x1_est[i] - x2_est[i], y1_est[i] - y2_est[i]])
        # for each x1_est find nearest ground truth point in x1
        dists = np.linalg.norm(np.vstack((x1_est[i] - x1, y1_est[i] - y1)), axis=0)
        best_matches = np.argsort(dists)

        # find the best match that is not taken yet
        for match_idx in best_matches:
            if not matched[match_idx]:
                break
        else:
            continue

        # A match is good only if
        # (1) An unmatched GT point exists within 150 pixels, and
        # (2) GT correspondence offset is within 25 pixels of estimated
        #     correspondence offset
        gt_offset = np.asarray([x1[match_idx] - x2[match_idx], y1[match_idx] - y2[match_idx]])
        offset_dist = np.linalg.norm(cur_offset - gt_offset)
        if (dists[match_idx] < 150.0) and (offset_dist < 25):
            good_matches[i] = True
            # pass #print('Correct')
        else:
            pass  # print('Incorrect')

    print("You found {}/{} required matches".format(num_matches, num_req_matches))
    accuracy = np.mean(good_matches) * min(num_matches, num_req_matches) * 1.0 / num_req_matches
    print("Accuracy = {:f}".format(accuracy))
    green = np.asarray([0, 1, 0], dtype=float)
    red = np.asarray([1, 0, 0], dtype=float)
    line_colors = np.asarray([green if m else red for m in good_matches])

    rendered_img = show_correspondence_lines(
        imgA,
        imgB,
        x1_est * scale_factor,
        y1_est * scale_factor,
        x2_est * scale_factor,
        y2_est * scale_factor,
        line_colors,
    )

    return accuracy, rendered_img


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    desc_dim = feature_width ** 2
    num_kps = X.shape[0]
    fvs = np.zeros((num_kps, desc_dim))
    radius = feature_width // 2

    for i, (x, y) in enumerate(zip(X, Y)):

        h_start = x - radius + 1
        h_end = x + radius + 1

        v_start = y - radius + 1
        v_end = y + radius + 1

        fvs[i] = image_bw[v_start:v_end, h_start:h_end].flatten()
        fvs[i] /= np.linalg.norm(fvs[i])

    return fvs


def evaluate_points(
    P: np.ndarray, points_2d: np.ndarray, points_3d: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Evaluate the residual between actual 2D points and the projected 2D
    points calculated from the projection matrix.

    You do not need to modify anything in this function, although you can if you
    want to.

    Args:
        M: a 3 x 4 numpy array representing the projection matrix.
        points_2d: a N x 2 numpy array representing the 2D points.
        points_3d: a N x 3 numpy array representing the 3D points.

    Returns:
        estimated_points_2d: a N x 2 numpy array representing the projected
            2D points
        residual: a float value representing the residual
    """

    estimated_points_2d = projection(P, points_3d)

    residual = np.mean(
        np.hypot(
            estimated_points_2d[:, 0] - points_2d[:, 0],
            estimated_points_2d[:, 1] - points_2d[:, 1],
        )
    )

    return estimated_points_2d, residual


def visualize_points_image(
    actual_pts: np.ndarray, projected_pts: np.ndarray, im_path: str
) -> None:
    """Visualize the actual 2D points and the projected 2D points calculated
    from the projection matrix.

    You do not need to modify anything in this function, although you can if
    you want to.

    Args:
        actual_pts: a N x 2 numpy array representing the actual 2D points.
        projected_pts: a N x 2 numpy array representing the projected 2D points.
        im_path: a string representing the path to the image.

    Returns:
        None
    """

    im = load_image(im_path)
    _, ax = plt.subplots()

    ax.imshow(im)
    ax.scatter(
        actual_pts[:, 0], actual_pts[:, 1], c="red", marker="o", label="Actual points"
    )
    ax.scatter(
        projected_pts[:, 0],
        projected_pts[:, 1],
        c="green",
        marker="+",
        label="Projected points",
    )

    ax.legend()


def visualize_points(actual_pts: np.ndarray, projected_pts: np.ndarray) -> None:
    """Visualize the actual 2D points and the projected 2D points calculated
    from the projection matrix.

    You do not need to modify anything in this function, although you can if
    you want to.

    Args:
        actual_pts: a N x 2 numpy array representing the actual 2D points.
        projected_pts: a N x 2 numpy array representing the projected 2D points.

    Returns:
        None
    """
    _, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        actual_pts[:, 0], actual_pts[:, 1], c="red", marker="o", label="Actual points"
    )
    ax.scatter(
        projected_pts[:, 0],
        projected_pts[:, 1],
        c="green",
        marker="+",
        label="Projected points",
    )

    plt.ylim(max(plt.ylim()), min(plt.ylim()))
    ax.legend()
    ax.axis("equal")


def plot3dview_2_cameras(
    points_3d: np.ndarray,
    camera_center_1: np.ndarray,
    camera_center_2: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
) -> None:
    """Visualize the actual 3D points and the estimated 3D camera center for
    2 cameras.

    You do not need to modify anything in this function, although you can if
    you want to.

    Args:
        points_3d: a N x 3 numpy array representing the actual 3D points
        camera_center_1: a 1 x 3 numpy array representing the first camera
            center
        camera_center_2: a 1 x 3 numpy array representing the second camera
            center
        R1: a 3 x 3 numpy array representing the rotation matrix for the first
            camera
        R2: a 3 x 3 numpy array representing the rotation matrix for the second
            camera

    Returns:
        None
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.scatter(
        points_3d[:, 0],
        points_3d[:, 1],
        points_3d[:, 2],
        c="blue",
        marker="o",
        s=10,
        depthshade=0,
    )

    camera_center_1 = camera_center_1.squeeze()
    ax.scatter(
        camera_center_1[0],
        camera_center_1[1],
        camera_center_1[2],
        c="red",
        marker="x",
        s=20,
        depthshade=0,
    )

    camera_center_2 = camera_center_2.squeeze()
    ax.scatter(
        camera_center_2[0],
        camera_center_2[1],
        camera_center_2[2],
        c="red",
        marker="x",
        s=20,
        depthshade=0,
    )

    v1 = R1[:, 0] * 5
    v2 = R1[:, 1] * 5
    v3 = R1[:, 2] * 5

    cc0, cc1, cc2 = camera_center_1

    ax.plot3D([0, 5], [0, 0], [0, 0], c="r")
    ax.plot3D([0, 0], [0, 5], [0, 0], c="g")
    ax.plot3D([0, 0], [0, 0], [0, 5], c="b")

    ax.plot3D([cc0, cc0 + v1[0]], [cc1, cc1 + v1[1]], [cc2, cc2 + v1[2]], c="r")
    ax.plot3D([cc0, cc0 + v2[0]], [cc1, cc1 + v2[1]], [cc2, cc2 + v2[2]], c="g")
    ax.plot3D([cc0, cc0 + v3[0]], [cc1, cc1 + v3[1]], [cc2, cc2 + v3[2]], c="b")

    v1 = R2[:, 0] * 5
    v2 = R2[:, 1] * 5
    v3 = R2[:, 2] * 5

    cc0, cc1, cc2 = camera_center_2

    ax.plot3D([0, 1], [0, 0], [0, 0], c="r")
    ax.plot3D([0, 0], [0, 1], [0, 0], c="g")
    ax.plot3D([0, 0], [0, 0], [0, 1], c="b")

    ax.plot3D([cc0, cc0 + v1[0]], [cc1, cc1 + v1[1]], [cc2, cc2 + v1[2]], c="r")
    ax.plot3D([cc0, cc0 + v2[0]], [cc1, cc1 + v2[1]], [cc2, cc2 + v2[2]], c="g")
    ax.plot3D([cc0, cc0 + v3[0]], [cc1, cc1 + v3[1]], [cc2, cc2 + v3[2]], c="b")

    # draw vertical lines connecting each point to ground
    min_z = min(points_3d[:, 2])
    for p in points_3d:
        x, y, z = p
        ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c="black", linewidth=1)

    x, y, z = camera_center_1
    ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c="black", linewidth=1)

    x, y, z = camera_center_2
    ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c="black", linewidth=1)


def plot3dview_with_coordinates(
    points_3d: np.ndarray, camera_center: np.ndarray, R: np.ndarray
) -> None:
    """Visualize the actual 3D points and the estimated 3D camera center.

    You do not need to modify anything in this function, although you can if
    you want to.

    Args:
        points_3d: a N x 3 numpy array representing the actual 3D points.
        camera_center: a 1 x 3 numpy array representing the camera center.
        R: a 3 x 3 numpy array representing the rotation matrix for the camera.

    Returns:
        None
    """

    v1 = R[:, 0] * 5
    v2 = R[:, 1] * 5
    v3 = R[:, 2] * 5

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.scatter(
        points_3d[:, 0],
        points_3d[:, 1],
        points_3d[:, 2],
        c="blue",
        marker="o",
        s=10,
        depthshade=0,
    )
    camera_center = camera_center.squeeze()
    ax.scatter(
        camera_center[0],
        camera_center[1],
        camera_center[2],
        c="red",
        marker="x",
        s=20,
        depthshade=0,
    )

    cc0, cc1, cc2 = camera_center

    ax.plot3D([0, 5], [0, 0], [0, 0], c="r")
    ax.plot3D([0, 0], [0, 5], [0, 0], c="g")
    ax.plot3D([0, 0], [0, 0], [0, 5], c="b")

    ax.plot3D([cc0, cc0 + v1[0]], [cc1, cc1 + v1[1]], [cc2, cc2 + v1[2]], c="r")
    ax.plot3D([cc0, cc0 + v2[0]], [cc1, cc1 + v2[1]], [cc2, cc2 + v2[2]], c="g")
    ax.plot3D([cc0, cc0 + v3[0]], [cc1, cc1 + v3[1]], [cc2, cc2 + v3[2]], c="b")

    # draw vertical lines connecting each point to ground
    min_z = min(points_3d[:, 2])
    for p in points_3d:
        x, y, z = p
        ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c="black", linewidth=1)
    x, y, z = camera_center
    ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c="black", linewidth=1)


def plot3dview(points_3d: np.ndarray, camera_center: np.ndarray) -> None:
    """
    Visualize the actual 3D points and the estimated 3D camera center.

    You do not need to modify anything in this function, although you can if
    you want to.

    Args:
        points_3d: a N x 3 numpy array representing the actual 3D points.
        camera_center: a 1 x 3 numpy array representing the camera center.

    Returns:
        None
    """
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.scatter(
        points_3d[:, 0],
        points_3d[:, 1],
        points_3d[:, 2],
        c="blue",
        marker="o",
        s=10,
        depthshade=0,
    )
    camera_center = camera_center.squeeze()
    ax.scatter(
        camera_center[0],
        camera_center[1],
        camera_center[2],
        c="red",
        marker="x",
        s=20,
        depthshade=0,
    )

    # draw vertical lines connecting each point to ground
    min_z = min(points_3d[:, 2])
    for p in points_3d:
        x, y, z = p
        ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c="black", linewidth=1)
    x, y, z = camera_center
    ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c="black", linewidth=1)

    set_axes_equal(ax)

    return ax


def draw_epipolar_lines(
    F: np.ndarray,
    img_left: np.ndarray,
    img_right: np.ndarray,
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    figsize=(10, 8),
):
    """Draw the epipolar lines given the fundamental matrix, left right images
    and left right datapoints

    You do not need to modify anything in this function.

    Args:
        F: a 3 x 3 numpy array representing the fundamental matrix, such that
            p_right^T @ F @ p_left = 0 for correct correspondences
        img_left: array representing image 1.
        img_right: array representing image 2.
        pts_left: array of shape (N,2) representing image 1 datapoints.
        pts_right: array of shape (N,2) representing image 2 datapoints.

    Returns:
        None
    """
    # ------------ lines in the RIGHT image --------------------
    imgh_right, imgw_right = img_right.shape[:2]
    # corner points, as homogeneous (x,y,1)
    p_ul = np.asarray([0, 0, 1])
    p_ur = np.asarray([imgw_right, 0, 1])
    p_bl = np.asarray([0, imgh_right, 1])
    p_br = np.asarray([imgw_right, imgh_right, 1])

    # The equation of the line through two points
    # can be determined by taking the ‘cross product’
    # of their homogeneous coordinates.

    # left and right border lines, for the right image
    l_l = np.cross(p_ul, p_bl)
    l_r = np.cross(p_ur, p_br)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    ax[1].imshow(img_right)
    ax[1].autoscale(False)
    ax[1].scatter(
        pts_right[:, 0], pts_right[:, 1], marker="o", s=20, c="yellow", edgecolors="red"
    )
    for p in pts_left:
        p = np.hstack((p, 1))[:, np.newaxis]
        # get defn of epipolar line in right image, corresponding to left point p
        l_e = np.dot(F, p).squeeze()
        # find where epipolar line in right image crosses the left and image borders
        p_l = np.cross(l_e, l_l)
        p_r = np.cross(l_e, l_r)
        # convert back from homogeneous to cartesian by dividing by 3rd entry
        # draw line between point on left border, and on the right border
        x = [p_l[0] / p_l[2], p_r[0] / p_r[2]]
        y = [p_l[1] / p_l[2], p_r[1] / p_r[2]]
        ax[1].plot(x, y, linewidth=1, c="blue")

    # ------------ lines in the LEFT image --------------------
    imgh_left, imgw_left = img_left.shape[:2]

    # corner points, as homogeneous (x,y,1)
    p_ul = np.asarray([0, 0, 1])
    p_ur = np.asarray([imgw_left, 0, 1])
    p_bl = np.asarray([0, imgh_left, 1])
    p_br = np.asarray([imgw_left, imgh_left, 1])

    # left and right border lines, for left image
    l_l = np.cross(p_ul, p_bl)
    l_r = np.cross(p_ur, p_br)

    ax[0].imshow(img_left)
    ax[0].autoscale(False)
    ax[0].scatter(
        pts_left[:, 0], pts_left[:, 1], marker="o", s=20, c="yellow", edgecolors="red"
    )
    for p in pts_right:
        p = np.hstack((p, 1))[:, np.newaxis]
        # defn of epipolar line in the left image, corresponding to point p in the right image
        l_e = np.dot(F.T, p).squeeze()
        p_l = np.cross(l_e, l_l)
        p_r = np.cross(l_e, l_r)
        x = [p_l[0] / p_l[2], p_r[0] / p_r[2]]
        y = [p_l[1] / p_l[2], p_r[1] / p_r[2]]
        ax[0].plot(x, y, linewidth=1, c="blue")


def get_matches(
    pic_a: np.ndarray, pic_b: np.ndarray, n_feat: int
) -> (np.ndarray, np.ndarray):
    """Get unreliable matching points between two images using SIFT.

    You do not need to modify anything in this function, although you can if
    you want to.

    Args:
        pic_a: a numpy array representing image 1.
        pic_b: a numpy array representing image 2.
        n_feat: an int representing number of matching points required.

    Returns:
        pts_a: a numpy array representing image 1 points.
        pts_b: a numpy array representing image 2 points.
    """
    pic_a = cv2.cvtColor(pic_a, cv2.COLOR_BGR2GRAY)
    pic_b = cv2.cvtColor(pic_b, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp_a, desc_a = sift.detectAndCompute(pic_a, None)
    kp_b, desc_b = sift.detectAndCompute(pic_b, None)
    dm = cv2.BFMatcher(cv2.NORM_L2)
    matches = dm.knnMatch(desc_b, desc_a, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < n.distance / 1.2:
            good_matches.append(m)
    pts_a = []
    pts_b = []
    for m in good_matches[: int(n_feat)]:
        pts_a.append(kp_a[m.trainIdx].pt)
        pts_b.append(kp_b[m.queryIdx].pt)

    return np.asarray(pts_a), np.asarray(pts_b)


def hstack_images(imgA: np.ndarray, imgB: np.ndarray) -> np.ndarray:
    """Stacks 2 images side-by-side

    Args:
        imgA: a numpy array representing image 1.
        imgB: a numpy array representing image 2.

    Returns:
        img: a numpy array representing the images stacked side by side.
    """
    Height = max(imgA.shape[0], imgB.shape[0])
    Width = imgA.shape[1] + imgB.shape[1]

    newImg = np.zeros((Height, Width, 3), dtype=imgA.dtype)
    newImg[: imgA.shape[0], : imgA.shape[1], :] = imgA
    newImg[: imgB.shape[0], imgA.shape[1] :, :] = imgB

    return newImg


def show_correspondence2(
    imgA: np.ndarray,
    imgB: np.ndarray,
    X1: np.ndarray,
    Y1: np.ndarray,
    X2: np.ndarray,
    Y2: np.ndarray,
    line_colors=None,
) -> None:
    """Visualizes corresponding points between two images. Corresponding points
    will have the same random color.

    Args:
        imgA: a numpy array representing image 1.
        imgB: a numpy array representing image 2.
        X1: a numpy array representing x coordinates of points from image 1.
        Y1: a numpy array representing y coordinates of points from image 1.
        X2: a numpy array representing x coordinates of points from image 2.
        Y2: a numpy array representing y coordinates of points from image 2.
        line_colors: a N x 3 numpy array containing colors of correspondence
            lines (optional)

    Returns:
        None
    """
    newImg = hstack_images(imgA, imgB)
    shiftX = imgA.shape[1]
    X1 = X1.astype(np.int32)
    Y1 = Y1.astype(np.int32)
    X2 = X2.astype(np.int32)
    Y2 = Y2.astype(np.int32)

    dot_colors = np.random.rand(len(X1), 3)
    if imgA.dtype == np.uint8:
        dot_colors *= 255
    if line_colors is None:
        line_colors = dot_colors

    for x1, y1, x2, y2, dot_color, line_color in zip(
        X1, Y1, X2, Y2, dot_colors, line_colors
    ):
        newImg = cv2.circle(newImg, (x1, y1), 5, dot_color, -1)
        newImg = cv2.circle(newImg, (x2 + shiftX, y2), 5, dot_color, -1)
        newImg = cv2.line(
            newImg, (x1, y1), (x2 + shiftX, y2), line_color, 2, cv2.LINE_AA
        )

    return newImg


def set_axes_equal(ax: Axes) -> None:
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Ref: https://github.com/borglab/gtsam/blob/develop/python/gtsam/utils/plot.py#L13

    Args:
        ax: axis for the plot.
    Returns:
        None
    """
    # get the min and max value for each of (x, y, z) axes as 3x2 matrix.
    # This gives us the bounds of the minimum volume cuboid encapsulating all
    # data.
    limits = np.array(
        [
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ]
    )

    # find the centroid of the cuboid
    centroid = np.mean(limits, axis=1)

    # pick the largest edge length for this cuboid
    largest_edge_length = np.max(np.abs(limits[:, 1] - limits[:, 0]))

    # set new limits to draw a cube using the largest edge length
    radius = 0.5 * largest_edge_length
    ax.set_xlim3d([centroid[0] - radius, centroid[0] + radius])
    ax.set_ylim3d([centroid[1] - radius, centroid[1] + radius])
    ax.set_zlim3d([centroid[2] - radius, centroid[2] + radius])
