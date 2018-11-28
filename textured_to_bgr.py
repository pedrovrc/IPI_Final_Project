import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt


# ---------- Start of functions ----------
def ycbcr2bgr(Y_img, Cb_img, Cr_img):   # Joins Y, Cb and Cr channels into BGR image
    # Gets dimensions from Y to create Red, Green and Blue component matrices
    # These matrices have dtype=np.float64 to reduce rounding errors
    height, width = Y_img.shape
    Red_img = np.zeros((height, width), dtype=np.float64)
    Green_img = np.zeros((height, width), dtype=np.float64)
    Blue_img = np.zeros((height, width), dtype=np.float64)

    # YCbCr to RGB conversion equations
    Red_img[:, :] = Y_img[:, :] + (1.402 * Cr_img[:, :] - 128 * 1.402)
    Green_img[:, :] = Y_img[:, :] - (0.344 * Cb_img[:, :] - 128 * 0.344) - (0.714 * Cr_img[:, :] - 128 * 0.714)
    Blue_img[:, :] = Y_img[:, :] + (1.772 * Cb_img[:, :] - 128 * 1.772)

    # Remaps values to uint8 scale [0, 255]
    Red_img = remap_tonescale(Red_img, 255, 0)
    Green_img = remap_tonescale(Green_img, 255, 0)
    Blue_img = remap_tonescale(Blue_img, 255, 0)

    # Concatenates BGR components into final image
    final_img = np.zeros((height, width, 3), dtype=np.uint8)
    final_img[:, :, 0] = Blue_img[:, :]
    final_img[:, :, 1] = Green_img[:, :]
    final_img[:, :, 2] = Red_img[:, :]
    return final_img


def remap_tonescale(image, high, low):  # Remaps an image's tone scale to chosen scale
    height, width = image.shape
    norm = np.zeros((height, width), dtype=np.float64)
    cv2.normalize(image, norm, alpha=low, beta=high, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    return norm
# ---------- End of Functions ----------


# ---------- Start of Main Program ----------
# Load and show textured image
tex_img = cv2.imread("texturizada.png", cv2.IMREAD_GRAYSCALE)
height, width = tex_img.shape
cv2.imshow("Textured", tex_img)
cv2.waitKey(0)

# Compute wavelet transform of textured image
cA, (cH, cV, cD) = pywt.dwt2(tex_img, 'haar')

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

# Get Cb and Cr channels from Horizontal and Vertical components
Cb_channel = cv2.resize(cH, (width, height), interpolation=cv2.INTER_CUBIC)
Cr_channel = cv2.resize(cV, (width, height), interpolation=cv2.INTER_CUBIC)

# Resize cA to use as Y'
#cA = pywt.idwt2((cA, (None, None, cD)), 'haar')
Y_channel = cv2.resize(cA, (width, height), interpolation=cv2.INTER_CUBIC)

# Convert Y', Cb' and Cr' channels to BGR components and show result
result = ycbcr2bgr(Y_channel, Cb_channel, Cr_channel)
cv2.imshow("Result", result)
cv2.waitKey(0)

# Save result image
cv2.imwrite("de_volta.png", result)
#cv2.imwrite("Y'.png", Y_channel)
#cv2.imwrite("Cb'.png", Cb_channel)
#cv2.imwrite("Cr'.png", Cr_channel)

cv2.destroyAllWindows()
# ---------- End of Main Program ----------
