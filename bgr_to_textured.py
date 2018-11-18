import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt


# ---------- Start of Functions ----------
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
# ---------- End of Functions ----------


# ---------- Start of Main Program ----------
# Load and show original bgr image
bgr_image = cv2.imread("mapa.png")
height, width, channels = bgr_image.shape
cv2.imshow("original", bgr_image)
cv2.waitKey(0)

# Create and show Y, Cb and Cr channels
Y_channel, Cb_channel, Cr_channel = bgr2ycbcr(bgr_image)
cv2.imshow("Y", Y_channel)
cv2.imshow("Cb", Cb_channel)
cv2.imshow("Cr", Cr_channel)
cv2.waitKey(0)

# Compute wavelet transform
coeffs = pywt.dwt2(Y_channel, 'db2')
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

# Replace cH and cV with Cb and Cr channels, respectively
new_Cb = cv2.resize(Cb_channel, (cH.shape[1], cH.shape[0]))
new_Cr = cv2.resize(Cr_channel, (cV.shape[1], cV.shape[0]))
cH[:, :] = new_Cb[:, :]
cV[:, :] = new_Cr[:, :]

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

# Compute inverse transform and show result
result = pywt.idwt2(coeffs, 'db2')
plt.imshow(result, interpolation="nearest", cmap=plt.cm.gray)
plt.show()

# Save result image
cv2.imwrite("texturizado.png", result)

cv2.destroyAllWindows()
# ---------- End of Main Program ----------
