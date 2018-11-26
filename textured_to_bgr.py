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
    #plt.imshow(Cb_img, interpolation="bicubic", cmap=plt.cm.gray)
    #plt.show()
    #plt.imshow(Cr_img, interpolation="bicubic", cmap=plt.cm.gray)
    #plt.show()

    # Remapping channels
    #Y_img = remap_tonescale(Y_img, 255, 0)
    #Cb_img = remap_tonescale(Cb_img, 255, 0)
    #Cr_img = remap_tonescale(Cr_img, 255, 0)

    # Show Cb and Cr after changes
    #plt.imshow(Cb_img, interpolation="bicubic", cmap=plt.cm.gray)
    #plt.show()
    #plt.imshow(Cr_img, interpolation="bicubic", cmap=plt.cm.gray)
    #plt.show()

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

    # Saving image components for analysis
    #low_offs = (Cb_img.min() * 255)/(Cb_img.max() - Cb_img.min())
    #high_offs = (Cb_img.max() * 255)/(Cb_img.max() - Cb_img.min())
    #Cb_img = remap_tonescale(Cb_img, high_offs, low_offs)
    #Cb_img = remap_tonescale(Cb_img, 255, 0)

    #low_offs = (Cr_img.min() * 255)/(Cr_img.max() - Cr_img.min())
    #high_offs = (Cr_img.max() * 255)/(Cr_img.max() - Cr_img.min())
    #Cr_img = remap_tonescale(Cr_img, high_offs, low_offs)
    #Cr_img = remap_tonescale(Cr_img, 255, 0)

    #cv2.imwrite("recovered_Cb.png", Cb_img.astype(np.uint8))
    #cv2.imwrite("recovered_Cr.png", Cr_img.astype(np.uint8))

    return final_img


# Remaps an image's tone scale to chosen scale
def remap_tonescale(image, high, low):
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