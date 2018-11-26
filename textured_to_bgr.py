import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt


# ---------- Start of functions ----------
# Join my images so as to have a single one
def ycbcr2bgr(Y_img, Cb_img, Cr_img):

    # Getting the sizes of my Y image so as to create Red, Green and Blue ones
    height, width = Y_img.shape
    Red_img = np.zeros((height, width), dtype=np.float64)
    Green_img = np.zeros((height, width), dtype=np.float64)
    Blue_img = np.zeros((height, width), dtype=np.float64)
    # Obs.: using dtype = np.float64 so as to determine which values will be negative,
    # and then remap the values according to np.uint8

    # Show Cb and Cr before changes
    plt.imshow(Cb_img, interpolation="bicubic", cmap=plt.cm.gray)
    plt.show()
    plt.imshow(Cr_img, interpolation="bicubic", cmap=plt.cm.gray)
    plt.show()

    # Remapping channels
    Y_img = remap_tonescale(Y_img, 255, 0)
    Cb_img = remap_tonescale(Cb_img, 255, 0)
    Cr_img = remap_tonescale(Cr_img, 255, 0)

    # Power laws to fix contrast
    # Cb_img = cv2.pow(Cb_img, 0.8)
    # Cr_img = cv2.pow(Cr_img, 0.8)

    # Show Cb and Cr after changes
    plt.imshow(Cb_img, interpolation="bicubic", cmap=plt.cm.gray)
    plt.show()
    plt.imshow(Cr_img, interpolation="bicubic", cmap=plt.cm.gray)
    plt.show()

    # Saving image components for analysis
    # cv2.imwrite("recovered_Cb.png", Cb_img)
    # cv2.imwrite("recovered_Cr.png", Cr_img)

    # Converting YCbCr image to Red, Green and Blue images
    Red_img[:, :] = Y_img[:, :] + (1.402 * Cr_img[:, :] - 128 * 1.402)
    Green_img[:, :] = Y_img[:, :] - (0.344 * Cb_img[:, :] - 128 * 0.344) - (0.714 * Cr_img[:, :] - 128 * 0.714)
    Blue_img[:, :] = Y_img[:, :] + (1.772 * Cb_img[:, :] - 128 * 1.772)

    # Remapping to uint8 scale [0, 255]
    Red_img = remap_tonescale(Red_img, 255, 0)
    Green_img = remap_tonescale(Green_img, 255, 0)
    Blue_img = remap_tonescale(Blue_img, 255, 0)

    # Putting together the Red, Green and Blue images so as to build a final RGB image
    final_img = np.zeros((height, width, 3), dtype=np.uint8)
    final_img[:, :, 0] = Blue_img[:, :]
    final_img[:, :, 1] = Green_img[:, :]
    final_img[:, :, 2] = Red_img[:, :]

    return final_img


# Remaps an image's tone scale to chosen scale
def remap_tonescale(image, high, low):
    height, width = image.shape

    norm = np.zeros((height, width), dtype=np.float64)
    cv2.normalize(image, norm, alpha=low, beta=high, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    return norm


def median_filter(image):  # Uses median to remove peaks of 255 or 0 in image
    height, width = image.shape

    # Create padding around image
    borderedImage = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # For each pixel, get it and its neighbours in an array and get the median value as the new value
    for i in range(0, height):
        for j in range(0, width):
            pixel_array = get_pixels(borderedImage, i+1, j+1)
            pixel_array.sort()
            length = pixel_array.__len__()
            image[i, j] = pixel_array[int(length/2)-1]


def get_pixels(image, i, j):  # Returns a pixel and its 24 neighbours in an array
    px1 = image[i, j]
    px2 = image[i-1, j-1]
    px3 = image[i, j-1]
    px4 = image[i+1, j-1]
    px5 = image[i+1, j]
    px6 = image[i+1, j+1]
    px7 = image[i, j+1]
    px8 = image[i-1, j+1]
    px9 = image[i-1, j]
    px10 = image[i-2, j-2]
    px11 = image[i-2, j-1]
    px12 = image[i-2, j]
    px13 = image[i-2, j+1]
    px14 = image[i-2, j+2]
    px15 = image[i-1, j+2]
    px16 = image[i, j+2]
    px17 = image[i+1, j+2]
    px18 = image[i+2, j+2]
    px19 = image[i+2, j+1]
    px20 = image[i+2, j]
    px21 = image[i+2, j-1]
    px22 = image[i+2, j-2]
    px23 = image[i+1, j-2]
    px24 = image[i, j-2]
    px25 = image[i-1, j-2]
    return [px1, px2, px3, px4, px5, px6, px7, px8, px9, px10,
            px11, px12, px13, px14, px15, px16, px17, px18, px19, px20,
            px21, px22, px23, px24, px25]
# ---------- End of Functions ----------


# ---------- Start of Main Program ----------
# Load and show textured image
tex_img = cv2.imread("texturizada.png", cv2.IMREAD_GRAYSCALE)
height, width = tex_img.shape
cv2.imshow("Textured", tex_img)
cv2.waitKey(0)

# Compute wavelet transform of textured image
coeffs = pywt.dwt2(tex_img, 'haar')
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

# Get Cb and Cr channels from Horizontal and Vertical components
Cb_channel = cv2.resize(cH, (width, height))
Cr_channel = cv2.resize(cV, (width, height))

# Resize cA to use as Y'
#cA = pywt.idwt2((cA, (None, None, cD)), 'haar')
Y_channel = cv2.resize(cA, (width, height), interpolation=cv2.INTER_CUBIC)

# Convert Y', Cb and Cr channels to BGR components and show result
result = ycbcr2bgr(Y_channel, Cb_channel, Cr_channel)
cv2.imshow("Result", result)
cv2.waitKey(0)

# Save result image
cv2.imwrite("de_volta.png", result)

cv2.destroyAllWindows()
# ---------- End of Main Program ----------