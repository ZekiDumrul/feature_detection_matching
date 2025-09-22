import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path, grayscale=True):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Gorsel yuklenemedi: {path}")
    return img


def rotate_image(img, angle, scale=1.0):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (w, h))


def resize_image(img, w=None, h=None):
    h, w = img.shape[:2]
    width = w
    height = h
    return cv2.resize(img, (width, height))


def harris_corners(img, blockSize=2, ksize=3, k=0.04, threshold=0.01):
    dst = cv2.cornerHarris(np.float32(img), blockSize, ksize, k)
    dst = cv2.dilate(dst, None)
    img_corners = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_corners[dst > threshold * dst.max()] = [0, 0, 255]
    return img_corners


def compute_features(img, method='SIFT'):
    if method.upper() == 'SIFT':
        detector = cv2.SIFT_create()
    elif method.upper() == 'ORB':
        detector = cv2.ORB_create()
    else:
        raise ValueError("Desteklenen metodlar: SIFT, ORB")
    kp, des = detector.detectAndCompute(img, None)
    return kp, des


def bf_match(des1, des2, method='SIFT', top_n=20):
    if method.upper() == 'SIFT':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif method.upper() == 'ORB':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:top_n]


def flann_match(des1, des2, method='SIFT', ratio=0.7, top_n=20):
    if method.upper() == 'SIFT':
        index_params = dict(algorithm=1, trees=5)
    elif method.upper() == 'ORB':
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    else:
        raise ValueError("Desteklenen metodlar: SIFT, ORB")
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m,n in matches if m.distance < ratio*n.distance]
    return good[:top_n]



def draw_matches(img1, kp1, img2, kp2, matches):
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)


def plot_images(images, titles, figsize=(18,12), cmap='gray'):
    import math
    n = len(images)
    rows = math.ceil(n/3)
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(rows, 3, i)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    img_path = r'your_image'
    
    img = load_image(img_path)
    img_rotated = rotate_image(img, 30)
    img_resized = resize_image(img, img.shape[1]//2, img.shape[0]//2)

    # Harris
    harris_rot = harris_corners(img_rotated)
    harris_res = harris_corners(resize_image(img_resized, img.shape[1], img.shape[0]))

    # SIFT and ORB
    kp_rot_sift, des_rot_sift = compute_features(img_rotated, 'SIFT')
    kp_res_sift, des_res_sift = compute_features(resize_image(img_resized, img.shape[1], img.shape[0]), 'SIFT')

    kp_rot_orb, des_rot_orb = compute_features(img_rotated, 'ORB')
    kp_res_orb, des_res_orb = compute_features(resize_image(img_resized, img.shape[1], img.shape[0]), 'ORB')

    # Matching
    matches_sift = bf_match(des_rot_sift, des_res_sift, 'SIFT')
    matches_orb = bf_match(des_rot_orb, des_res_orb, 'ORB')

    flann_sift = flann_match(des_rot_sift, des_res_sift, 'SIFT')
    flann_orb = flann_match(des_rot_orb, des_res_orb, 'ORB')

    # Draw matches
    img_matches_sift = draw_matches(img_rotated, kp_rot_sift, resize_image(img_resized, img.shape[1], img.shape[0]), kp_res_sift, matches_sift)
    img_matches_orb = draw_matches(img_rotated, kp_rot_orb, resize_image(img_resized, img.shape[1], img.shape[0]), kp_res_orb, matches_orb)
    img_flann_sift = draw_matches(img_rotated, kp_rot_sift, resize_image(img_resized, img.shape[1], img.shape[0]), kp_res_sift, flann_sift)
    img_flann_orb = draw_matches(img_rotated, kp_rot_orb, resize_image(img_resized, img.shape[1], img.shape[0]), kp_res_orb, flann_orb)

    # Plot
    plot_images(
        [img_rotated, img_resized, harris_rot, harris_res, img_matches_sift, img_matches_orb, img_flann_sift, img_flann_orb],
        ['Rotated', 'Resized', 'Harris Rotated', 'Harris Resized', 'SIFT BF', 'ORB BF', 'SIFT FLANN', 'ORB FLANN']
    )

if __name__ == "__main__":
    main()

