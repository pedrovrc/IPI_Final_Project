import cv2
import numpy as np
import matplotlib.pyplot as plt

orig_Cb = cv2.imread("original_Cb.png", cv2.IMREAD_GRAYSCALE)
orig_Cr = cv2.imread("original_Cr.png", cv2.IMREAD_GRAYSCALE)

recov_Cb = cv2.imread("recovered_Cb.png", cv2.IMREAD_GRAYSCALE)
recov_Cr = cv2.imread("recovered_Cr.png", cv2.IMREAD_GRAYSCALE)

titles = ['Original', 'Recovered',
          'Original Histogram', 'Recovered Histogram']
fig = plt.figure(figsize=(12, 3))

ax = fig.add_subplot(1, 4, 1)
ax.imshow(orig_Cb, interpolation="nearest")
ax.set_title(titles[0], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(1, 4, 2)
ax.imshow(recov_Cb, interpolation="nearest")
ax.set_title(titles[1], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(1, 4, 3)
ax.hist(orig_Cb.ravel(), 256, [0, 256])
ax.set_title(titles[2], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(1, 4, 4)
ax.hist(recov_Cb.ravel(), 256, [0, 256])
ax.set_title(titles[3], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()

titles = ['Original', 'Recovered',
          'Original Histogram', 'Recovered Histogram']
fig2 = plt.figure(figsize=(12, 3))

ax = fig2.add_subplot(1, 4, 1)
ax.imshow(orig_Cr, interpolation="nearest")
ax.set_title(titles[0], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

ax = fig2.add_subplot(1, 4, 2)
ax.imshow(recov_Cr, interpolation="nearest")
ax.set_title(titles[1], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

ax = fig2.add_subplot(1, 4, 3)
ax.hist(orig_Cr.ravel(), 256, [0, 256])
ax.set_title(titles[2], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

ax = fig2.add_subplot(1, 4, 4)
ax.hist(recov_Cr.ravel(), 256, [0, 256])
ax.set_title(titles[3], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

fig2.tight_layout()
plt.show()
