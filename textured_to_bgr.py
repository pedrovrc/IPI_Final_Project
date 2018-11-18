import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt


# ---------- Start of functions ----------
# Join my images so as to have a single denoised one
def ycbcr2bgr(Y_img, Cb_img, Cr_img):

    # Getting the sizes of my Y image so as to create Red, Green and Blue ones
    height, width = Y_img.shape
    Red_img = np.zeros((height, width), dtype=np.int16)
    Green_img = np.zeros((height, width), dtype=np.int16)
    Blue_img = np.zeros((height, width), dtype=np.int16)
    # Obs.: using dtype = np.int16 so as to determine which values will be negative,
    # and then convert these values to 0

    # Converting my YCbCr image to Red, Green and Blue images
    Red_img[:, :] = Y_img[:, :] + (1.402 * Cr_img[:, :] - 128 * 1.402)
    Green_img[:, :] = Y_img[:, :] - (0.344 * Cb_img[:, :] - 128 * 0.344) - (0.714 * Cr_img[:, :] - 128 * 0.714136)
    Blue_img[:, :] = Y_img[:, :] + (1.772 * Cb_img[:, :] - 128 * 1.772)

    remap_tonescale(Cb_img)
    remap_tonescale(Cr_img)

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
    final_img = np.zeros((height, width, 3), dtype=np.uint8)
    final_img[:, :, 0] = Blue_img[:, :]
    final_img[:, :, 1] = Green_img[:, :]
    final_img[:, :, 2] = Red_img[:, :]

    # Converting my matrix to an unsigned one so as to build correctly the final image
    return final_img.astype(np.uint8)

def remap_tonescale(image):
    height, width = image.shape
    maximum = -50
    minimum = 100

    for i in range(0, height):
        for j in range(0, width):
            if (image[i, j] > maximum):
                maximum = image[i, j]
            if (image[i, j] < minimum):
                minimum = image[i, j]

    print(maximum)
    print(minimum)


# ---------- End of Functions ----------


# ---------- Start of Main Program ----------
# Load and show textured image
tex_img = cv2.imread("texturizada.png")
height, width, channels = tex_img.shape
cv2.imshow("Textured", tex_img)
cv2.waitKey(0)

# Compute wavelet transform of textured image
coeffs = pywt.dwt2(tex_img[:, :, 0], 'db2')
cA, (cH, cV, cD) = coeffs

# Get Cb and Cr channels from Horizontal and Vertical components
Cb_channel = cv2.resize(cH, (width, height))
Cr_channel = cv2.resize(cV, (width, height))

# Inverse transform without cH and cV to obtain Y' channel
Y_channel = pywt.idwt2((cA, (None, None, cD)), 'db2')
plt.imshow(Y_channel, interpolation="nearest", cmap=plt.cm.gray)
plt.show()

# Convert Y', Cb and Cr channels to BGR components and show result
result = ycbcr2bgr(Y_channel, Cb_channel, Cr_channel)
cv2.imshow("Result", result)
cv2.waitKey(0)

# Save result image
cv2.imwrite("de_volta.png", result)

cv2.destroyAllWindows()
# ---------- End of Main Program ----------
