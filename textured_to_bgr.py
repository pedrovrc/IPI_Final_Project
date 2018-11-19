import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
import math


# ---------- Start of functions ----------
# Join my images so as to have a single denoised one
def ycbcr2bgr(Y_img, Cb_img, Cr_img):

    # Getting the sizes of my Y image so as to create Red, Green and Blue ones
    height, width = Y_img.shape
    Red_img = np.zeros((height, width), dtype=np.int32)
    Green_img = np.zeros((height, width), dtype=np.int32)
    Blue_img = np.zeros((height, width), dtype=np.int32)
    # Obs.: using dtype = np.int16 so as to determine which values will be negative,
    # and then convert these values to 0

    new_Y = remap_tonescale(Y_img)
    new_Cb = remap_tonescale(Cb_img)
    new_Cr = remap_tonescale(Cr_img)

    cv2.imshow("Cb", new_Cb)
    cv2.imshow("Cr", new_Cr)
    cv2.waitKey(0)

    # Converting my YCbCr image to Red, Green and Blue images
    Red_img[:, :] = new_Y[:, :] + (1.402 * new_Cr[:, :] - 128 * 1.402)
    Green_img[:, :] = new_Y[:, :] - (0.344 * new_Cb[:, :] - 128 * 0.344) - (0.714 * new_Cr[:, :] - 128 * 0.714)
    Blue_img[:, :] = new_Y[:, :] + (1.772 * new_Cb[:, :] - 128 * 1.772)

    # Checking if there is any pixel out of the range (0-256)
    for line in range(0, height):
        for column in range(0, width):
            if Blue_img[line, column] > 255:
                Blue_img[line, column] = 255
            elif Blue_img[line, column] < 0:
                Blue_img[line, column] = 0

            if Green_img[line, column] > 255:
                Green_img[line, column] = 255
            elif Green_img[line, column] < 0:
                Green_img[line, column] = 0

            if Red_img[line, column] > 255:
                Red_img[line, column] = 255
            elif Red_img[line, column] < 0:
                Red_img[line, column] = 0

    # Putting together the Red, Green and Blue images so as to build a final RGB image
    final_img = np.zeros((height, width, 3), dtype=np.int32)
    final_img[:, :, 0] = Blue_img[:, :]
    final_img[:, :, 1] = Green_img[:, :]
    final_img[:, :, 2] = Red_img[:, :]

    # Converting my matrix to an unsigned one so as to build correctly the final image
    return final_img.astype(np.uint8)

def remap_tonescale(image):
    height, width = image.shape
    maximum = 0
    minimum = 0
    image2 = np.zeros((height, width), dtype=np.uint8)

    image2[:, :] = image[:, :]

    for i in range(0, height):
        for j in range(0, width):
            if (image[i, j] > maximum):
                maximum = image[i, j]
            if (image[i, j] < minimum):
                minimum = image[i, j]

    for i in range(0, height):
        for j in range(0, width):
            image[i, j] = image[i, j] - minimum

    maximum = maximum - minimum

    for i in range(0, height):
        for j in range(0, width):
            image2[i, j] = ((image[i, j]*255)/maximum)

    return image2
# ---------- End of Functions ----------


# ---------- Start of Main Program ----------
# Load and show textured image
tex_img = cv2.imread("texturizada.png")
height, width, channels = tex_img.shape
cv2.imshow("Textured", tex_img)
cv2.waitKey(0)

# Compute wavelet transform of textured image
coeffs = pywt.dwt2(tex_img[:, :, 0], 'haar')
cA, (cH, cV, cD) = coeffs

# Show components of wavelet transform
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([cA, cH, cV, cD]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.show()

# Get Cb and Cr channels from Horizontal and Vertical components
Cb_channel = cv2.resize(cH, (width, height))
Cr_channel = cv2.resize(cV, (width, height))

# Resize cA to use as Y'
cA_resize = cv2.resize(cA, (width, height))

# Convert Y', Cb and Cr channels to BGR components and show result
result = ycbcr2bgr(cA_resize, Cb_channel, Cr_channel)
cv2.imshow("Result", result)
cv2.waitKey(0)

# Save result image
cv2.imwrite("de_volta.png", result)

cv2.destroyAllWindows()
# ---------- End of Main Program ----------
