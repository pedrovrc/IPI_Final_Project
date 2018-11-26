import cv2
import matplotlib.pyplot as plt

# This program shows 4 images and their respective histograms for analysis purposes

# Read images to compare
orig_Cb = cv2.imread("original_Cb.png", cv2.IMREAD_GRAYSCALE)
orig_Cr = cv2.imread("original_Cr.png", cv2.IMREAD_GRAYSCALE)
recov_Cb = cv2.imread("recovered_Cb.png", cv2.IMREAD_GRAYSCALE)
recov_Cr = cv2.imread("recovered_Cr.png", cv2.IMREAD_GRAYSCALE)

# Create figure variable
titles = ['Original', 'Recovered',
          'Original Histogram', 'Recovered Histogram']
fig = plt.figure(figsize=(12, 3))

# Add image1
ax = fig.add_subplot(1, 4, 1)
ax.imshow(orig_Cb, interpolation="nearest")
ax.set_title(titles[0], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

# Add image2
ax = fig.add_subplot(1, 4, 2)
ax.imshow(recov_Cb, interpolation="nearest")
ax.set_title(titles[1], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

# Add image1's histogram
ax = fig.add_subplot(1, 4, 3)
ax.hist(orig_Cb.ravel(), 256, [0, 256])
ax.set_title(titles[2], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

# Add image2's histogram
ax = fig.add_subplot(1, 4, 4)
ax.hist(recov_Cb.ravel(), 256, [0, 256])
ax.set_title(titles[3], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

# End figure
fig.tight_layout()

# Create another figure
fig2 = plt.figure(figsize=(12, 3))

# Add image1
ax = fig2.add_subplot(1, 4, 1)
ax.imshow(orig_Cr, interpolation="nearest")
ax.set_title(titles[0], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

# Add image2
ax = fig2.add_subplot(1, 4, 2)
ax.imshow(recov_Cr, interpolation="nearest")
ax.set_title(titles[1], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

# Add image1's histogram
ax = fig2.add_subplot(1, 4, 3)
ax.hist(orig_Cr.ravel(), 256, [0, 256])
ax.set_title(titles[2], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

# Add image2's histogram
ax = fig2.add_subplot(1, 4, 4)
ax.hist(recov_Cr.ravel(), 256, [0, 256])
ax.set_title(titles[3], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

# End figure
fig2.tight_layout()

# Show figure
plt.show()
