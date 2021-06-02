import cv2
import numpy as np
from matplotlib import pyplot as plt
from m2bk import *
import sys


def get_pose(image1, image2, depth, k, display=True):
    kp1, des1 = extract_features(image1)
    if display == True:
        print("Number of features detected in frame {0}\n".format(len(kp1)))
        print("Coordinates of the first keypoint in frame {0}".format(str(kp1[0].pt)))
        visualize_features(image1, kp1)
    kp2, des2 = extract_features(image2)
    match = match_features(des1, des2)
    if display == True:
        print("Number of features matched in frame: {0}".format(len(match)))
    filtred_match = filter_matches_distance(match)
    if display == True:
        print("Number of filtred features match in frames: {0}".format(len(filtred_match)))
        visualize_matches(image1, kp1, image2, kp2, filtred_match)
    rmat, tvec = estimate_motion(filtred_match, kp1, kp2, k, depth1=depth)
    print(tvec)
    if display == True:
        print("Estimated rotation:\n {0}".format(rmat))
        print("Estimated translation:\n {0}".format(tvec))
    return rmat, tvec

def extract_features(image):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=3000)

    # Find the keypoints and descriptors with ORB
    kp, des = orb.detectAndCompute(image, None)

    return kp, des

def visualize_features(image, kp):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints

    Returns:
    """
    display = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(display)
    plt.show()

def match_features(des1, des2):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image

    Returns:
    match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor
    """
    # Define FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)

    # Initiate FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find matches with FLANN
    match = flann.knnMatch(des1, des2, k=2)

    return match

def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)
    plt.show()

def estimate_motion(match, kp1, kp2, k, depth1=None):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera calibration matrix

    Optional arguments:
    depth1 -- a depth map of the first frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    rmat -- recovered 3x3 rotation numpy matrix
    tvec -- recovered 3x1 translation numpy vector
    image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are
                     coordinates of the i-th match in the image coordinate system
    image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are
                     coordinates of the i-th match in the image coordinate system

    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []

    objectpoints = []

    # Iterate through the matched features
    for m in match:
        # Get the pixel coordinates of features f[k - 1] and f[k]
        u1, v1 = kp1[m.queryIdx].pt
        u2, v2 = kp2[m.trainIdx].pt

        # Get the scale of features f[k - 1] from the depth map
        s = depth1[int(v1), int(u1)]

        # Check for valid scale values
        if s < 1000:
            # Transform pixel coordinates to camera coordinates using the pinhole camera model
            p_c = np.linalg.inv(k) @ (s * np.array([u1, v1, 1]))

            # Save the results
            image1_points.append([u1, v1])
            image2_points.append([u2, v2])
            objectpoints.append(p_c)

    # Convert lists to numpy arrays
    objectpoints = np.vstack(objectpoints)
    imagepoints = np.array(image2_points)

    # Determine the camera pose from the Perspective-n-Point solution using the RANSAC scheme
    _, rvec, tvec, _ = cv2.solvePnPRansac(objectpoints, imagepoints, k, None)

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)
    return rmat, tvec

def filter_matches_distance(match, dist_threshold=0.6):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0)

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = [m for m, n in match if m.distance < (dist_threshold * n.distance)]

    return filtered_match

