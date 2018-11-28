import cv2
import matplotlib.pyplot as plt

def plot(orig, recov):
    titles = ['Original', 'Recovered',
              'Original Histogram', 'Recovered Histogram']
    fig = plt.figure(figsize=(12, 3))

    # Add image1
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(orig, interpolation="nearest")
    ax.set_title(titles[0], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add image2
    ax = fig.add_subplot(1, 4, 2)
    ax.imshow(recov, interpolation="nearest")
    ax.set_title(titles[1], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add image1's histogram
    ax = fig.add_subplot(1, 4, 3)
    ax.hist(orig.ravel(), 256, [0, 256])
    ax.set_title(titles[2], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add image2's histogram
    ax = fig.add_subplot(1, 4, 4)
    ax.hist(recov.ravel(), 256, [0, 256])
    ax.set_title(titles[3], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # End figure
    fig.tight_layout()

# This program shows 4 images and their respective histograms for analysis purposes

# Read images to compare
origY = cv2.imread("Y.png", cv2.IMREAD_GRAYSCALE)
origCb = cv2.imread("Cb.png", cv2.IMREAD_GRAYSCALE)
origCr = cv2.imread("Cr.png", cv2.IMREAD_GRAYSCALE)
recovY = cv2.imread("Y'.png", cv2.IMREAD_GRAYSCALE)
recovCb = cv2.imread("Cb'.png", cv2.IMREAD_GRAYSCALE)
recovCr = cv2.imread("Cr'.png", cv2.IMREAD_GRAYSCALE)

plot(origY, recovY)
plot(origCb, recovCb)
plot(origCr, recovCr)

# Show figure
plt.show()
