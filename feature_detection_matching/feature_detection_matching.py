import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gorsel yukleme
img_path = r'your_image'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Gorsel yuklenemedi! ")
    exit()

#gorsel boyutlarý
(h, w) = img.shape
center = (w // 2, h // 2)

# gorseli dondur 30 derece
M = cv2.getRotationMatrix2D(center, 30, 1.0)
img_rotated = cv2.warpAffine(img, M, (w, h))


img_resized = cv2.resize(img, (w // 2, h // 2))


# Harris Corner Detection
def harris_corners(image):
    dst = cv2.cornerHarris(np.float32(image), blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    img_corners = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img_corners[dst > 0.01 * dst.max()] = [0, 0, 255]
    return img_corners

img_rotated_harris = harris_corners(img_rotated)
img_resized_harris = harris_corners(cv2.resize(img_resized, (w, h)))


# Ozellik tespiti (SIFT / ORB)
# SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img_rotated, None)
kp2, des2 = sift.detectAndCompute(cv2.resize(img_resized, (w, h)), None)

# ORB
orb = cv2.ORB_create()
kp1_orb, des1_orb = orb.detectAndCompute(img_rotated, None)
kp2_orb, des2_orb = orb.detectAndCompute(cv2.resize(img_resized, (w, h)), None)


# BFMatcher
bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_sift = bf_sift.match(des1, des2)
matches_sift = sorted(matches_sift, key=lambda x: x.distance)
img_matches_sift = cv2.drawMatches(img_rotated, kp1, cv2.resize(img_resized, (w, h)), kp2, matches_sift[:20], None, flags=2)

bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(des1_orb, des2_orb)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)
img_matches_orb = cv2.drawMatches(img_rotated, kp1_orb, cv2.resize(img_resized, (w, h)), kp2_orb, matches_orb[:20], None, flags=2)


# FLANN Matcher
# SIFT
FLANN_INDEX_KDTREE = 1
index_params_sift = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann_sift = cv2.FlannBasedMatcher(index_params_sift, search_params)
matches_flann_sift = flann_sift.knnMatch(des1, des2, k=2)

# ratio test
good_matches_sift = []
for m,n in matches_flann_sift:
    if m.distance < 0.7 * n.distance:
        good_matches_sift.append(m)
img_flann_sift = cv2.drawMatches(img_rotated, kp1, cv2.resize(img_resized, (w, h)), kp2, good_matches_sift[:20], None, flags=2)

# ORB (binary descriptors için LSH)
FLANN_INDEX_LSH = 6
index_params_orb = dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 1)
flann_orb = cv2.FlannBasedMatcher(index_params_orb, search_params)
matches_flann_orb = flann_orb.knnMatch(des1_orb, des2_orb, k=2)

# ratio test
good_matches_orb = []
for m,n in matches_flann_orb:
    if m.distance < 0.7 * n.distance:
        good_matches_orb.append(m)
img_flann_orb = cv2.drawMatches(img_rotated, kp1_orb, cv2.resize(img_resized, (w, h)), kp2_orb, good_matches_orb[:20], None, flags=2)


# Sonuclari goster
plt.figure(figsize=(18,12))

plt.subplot(3,3,1), plt.imshow(img_rotated, cmap='gray'), plt.title('Rotated Image')
plt.subplot(3,3,2), plt.imshow(img_resized, cmap='gray'), plt.title('Resized Image')

plt.subplot(3,3,3), plt.imshow(img_rotated_harris), plt.title('Harris Corners Rotated')
plt.subplot(3,3,4), plt.imshow(img_resized_harris), plt.title('Harris Corners Resized')

plt.subplot(3,3,5), plt.imshow(img_matches_sift), plt.title('SIFT BFMatcher')
plt.subplot(3,3,6), plt.imshow(img_matches_orb), plt.title('ORB BFMatcher')

plt.subplot(3,3,7), plt.imshow(img_flann_sift), plt.title('SIFT FLANN')
plt.subplot(3,3,8), plt.imshow(img_flann_orb), plt.title('ORB FLANN')

plt.tight_layout()
plt.show()
