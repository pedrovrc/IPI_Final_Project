import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt


# ---------- Start of Functions ----------
def bgr2ycbcr(image):  # Converts image from bgr space to ycbcr space
    height, width, channels = image.shape

    Y = np.zeros((height, width), dtype=np.float64)
    Y[:, :] += image[:, :, 2] * 0.299
    Y[:, :] += image[:, :, 1] * 0.587
    Y[:, :] += image[:, :, 0] * 0.114

    Cb = np.zeros((height, width), dtype=np.float64)
    Cr = np.zeros((height, width), dtype=np.float64)

    # Other channels derived from Y channel, blue and red
    Cb[:, :] = (0.564 * image[:, :, 0]) - (0.564 * Y[:, :]) + 128
    Cr[:, :] = (0.713 * image[:, :, 2]) - (0.713 * Y[:, :]) + 128

    return [Y, Cb, Cr]


def remap_tonescale(image, high, low):  # Used to stretch Cb and Cr mappings to fit cH and cV scales
    height, width = image.shape

    norm = np.zeros((height, width), dtype=np.float64)
    cv2.normalize(image, norm, alpha=low, beta=high, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    return norm
# ---------- End of Functions ----------


# ---------- Start of Main Program ----------
# Load and show original bgr image
bgr_image = cv2.imread("lena.jpg")
height, width, channels = bgr_image.shape
cv2.imshow("original", bgr_image)
cv2.waitKey(0)

# Create and save Y, Cb and Cr channels (saving for analysis)
Y_channel, Cb_channel, Cr_channel = bgr2ycbcr(bgr_image)
# cv2.imwrite("original_Cb.png", Cb_channel)
# cv2.imwrite("original_Cr.png", Cr_channel)

# Compute wavelet transform
coeffs = pywt.dwt2(Y_channel, 'haar')
cA, (cH, cV, cD) = coeffs

# Show components of wavelet transform
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([cA, cH, cV, cD]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="bicubic", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.show()

# Remap Cb and Cr channels to the cH and cV scales
Cb_channel = remap_tonescale(Cb_channel, cH.max(), cH.min())
Cr_channel = remap_tonescale(Cr_channel, cV.max(), cV.min())

# Replace cH and cV with Cb and Cr channels, respectively
cH = cv2.resize(Cb_channel, (cH.shape[1], cH.shape[0]))
cV = cv2.resize(Cr_channel, (cV.shape[1], cV.shape[0]))

# Show components of wavelet transform after inclusion of Cb and Cr
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([cA, cH, cV, cD]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="bicubic", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.show()

# Compute inverse transform and show result
result = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
plt.imshow(result, interpolation="bicubic", cmap=plt.cm.gray)
plt.show()

# Save result image
cv2.imwrite("texturizada.png", result)

cv2.destroyAllWindows()
# ---------- End of Main Program ----------
