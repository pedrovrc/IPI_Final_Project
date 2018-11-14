import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt


def bgr2ycbcr(image):  # Converts image from bgr space to ycbcr space
    height, width, channels = image.shape

    Y = np.zeros((height, width), dtype=np.uint8)
    Y_double = np.zeros((height, width), dtype=np.double)

    # Y_double made to facilitate fast processing while avoiding rounding errors
    Y_double[:, :] += image[:, :, 2] * 0.299
    Y_double[:, :] += image[:, :, 1] * 0.587
    Y_double[:, :] += image[:, :, 0] * 0.114

    Y[:, :] = Y_double[:, :]

    Cb = np.zeros((height, width), dtype=np.uint8)
    Cr = np.zeros((height, width), dtype=np.uint8)

    # Other channels derived from Y channel, blue and red
    Cb[:, :] = (0.564 * image[:, :, 0]) - (0.564 * Y[:, :]) + 128
    Cr[:, :] = (0.713 * image[:, :, 2]) - (0.713 * Y[:, :]) + 128

    return [Y, Cb, Cr]


def ycbcr2bgr(Y, Cb, Cr):  # Converts image from ycbcr space to bgr space
    height, width = Y.shape
    red = np.zeros((height, width), dtype=np.uint8)
    green = np.zeros((height, width), dtype=np.uint8)
    blue = np.zeros((height, width), dtype=np.uint8)

    # bgr channels are derived from ycbcr channels
    blue[:, :] = Y[:, :] + ((1.772 * Cb[:, :]) - (128 * 1.772))
    green[:, :] = Y[:, :] - ((0.344136 * Cb[:, :]) - (128 * 0.344136)) - ((0.714136 * Cr[:, :]) - (128 * 0.714136))
    red[:, :] = Y[:, :] + ((1.402 * Cr[:, :]) - (128 * 1.402))

    # Channels are concatenated into final product image
    result = np.zeros((height, width, 3), dtype=np.uint8)
    result[:, :, 0] = blue[:, :]
    result[:, :, 1] = green[:, :]
    result[:, :, 2] = red[:, :]

    return result


bgr_image = cv2.imread("mapa.png")
height, width, channels = bgr_image.shape
cv2.imshow("original", bgr_image)
cv2.waitKey(0)

Y_channel, Cb_channel, Cr_channel = bgr2ycbcr(bgr_image)

cv2.imshow("Y", Y_channel)
cv2.imshow("Cb", Cb_channel)
cv2.imshow("Cr", Cr_channel)
cv2.waitKey(0)

coeffs = pywt.dwt2(bgr_image, 'bior1.3')
LL, (LH, HL, HH) = coeffs

titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']

fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

cv2.destroyAllWindows()
