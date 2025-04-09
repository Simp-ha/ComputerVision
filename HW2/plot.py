import cv2
import numpy as np
import glob
from scipy.stats import entropy
import matplotlib.pylab as plot
Plot = False

folders = ['GES-50', 'NISwGSP', 'OpenPano/flower', 'mine']
i = input(f"Please write which panorama do you want\n1)GES\n2)NIS\n3)Flower\n4)Mine\nYour Answer is: ")

# Importing every source photo
image_path = glob.glob(folders[int(i)-1] + "/*.jpg")
s_img = [cv2.imread(image) for image in image_path]

# Importing every panorama result
image_path = glob.glob(f"{i}/*.jpg")
p_img = [cv2.imread(image) for image in image_path]
# Categorized in SIFT and SURF
s = len(p_img)//2
sift_imgs = p_img[:2]
surf_imgs = p_img[s:s+2]


def calculating_entropy(gImg, name):
    _bins = 128
    hist, _ = np.histogram(np.ravel(gImg), bins=_bins, range=(0, _bins))
    prob_dist = hist / hist.sum()
    if Plot:
        print(f"Plot of entropy histogram for photo {j}")
        plot.hist(hist, density=1, bins=_bins)
        plot.savefig('Fig_' + name)
        plot.show()
    return entropy(prob_dist, base=2)


# Slow
def calculating_local_entropy(gImg, sw_size=9):
    rows, cols = gImg.shape[:2]
    local_entropies = []

    half_window = sw_size // 2
    padded_image = cv2.copyMakeBorder(gImg, half_window, half_window, half_window, half_window, cv2.BORDER_REFLECT)
    _bins = 128
    for i in range(rows):
        for j in range(cols):
            window = padded_image[i:i + sw_size, j:j + sw_size]
            hist, _ = np.histogram(window.ravel(), bins=_bins, range=(0, _bins))
            prob_dist = hist / hist.sum()
            local_entropies.append(entropy(prob_dist))

    return np.array(local_entropies)


# Chatbased
def sliding_window_view(image, window_shape):
    """Manually implement sliding window view."""
    stride = image.strides
    shape = (
        image.shape[0] - window_shape[0] + 1,
        image.shape[1] - window_shape[1] + 1,
        window_shape[0],
        window_shape[1],
    )
    strides = stride + stride
    return np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)


def calculate_local_entropy_optimized(image, window_size=9):
    """Calculate local entropy for an image using a sliding window, optimized with a custom function."""
    rows, cols = image.shape
    half_window = window_size // 2

    # Pad the image
    padded_image = cv2.copyMakeBorder(image, half_window, half_window, half_window, half_window, cv2.BORDER_REFLECT)

    # Create sliding windows manually
    strided_windows = sliding_window_view(padded_image, (window_size, window_size))
    # strided_windows = strided_windows.reshape(rows, cols, -1)  # Flatten the window dimensions
    strided_windows = strided_windows.reshape(cols, rows, -1) # Flatten the window dimensions

    # Compute histograms for all windows simultaneously
    histograms = np.apply_along_axis(
        lambda w: np.histogram(w, bins=256, range=(0, 256), density=True)[0],
        axis=2,
        arr=strided_windows
    )

    # Compute entropies for each window
    local_entropies = np.apply_along_axis(entropy, axis=2, arr=histograms)

    return local_entropies.ravel()  # Flatten to return a 1D array


j = 2
s_img = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in s_img]
source = [s_img[i] for i in range(2*j, 2*j+1)]

panorama_surf = cv2.cvtColor(surf_imgs[j], cv2.COLOR_BGR2GRAY)
panorama_sift = cv2.cvtColor(sift_imgs[j], cv2.COLOR_BGR2GRAY)

# Global entropy
source_entropy = [calculating_entropy(img, f"Source{j}") for j, img in enumerate(source)]
panorama_entropy_SURF = calculating_entropy(panorama_surf, "SURF")
panorama_entropy_SIFT = calculating_entropy(panorama_sift, "SIFT")

# Local entropy
source_local_entropy = [calculate_local_entropy_optimized(img) for j, img in enumerate(source)]
panorama_local_entropy_SURF = calculate_local_entropy_optimized(panorama_surf)
panorama_local_entropy_SIFT = calculate_local_entropy_optimized(panorama_sift)

# Standard deviation
std_source = [np.std(img) for img in source]
std_panorama_surf = np.std(panorama_surf)
std_panorama_sift = np.std(panorama_sift)

surff2 = panorama_entropy_SURF - np.mean(source_entropy)
siftf2 = panorama_entropy_SIFT - np.mean(source_entropy)
print(f"(differential entropy SURF) f2 = {surff2}")
print(f"(differential entropy SIFT) f2 = {siftf2}")

f3 = np.mean(source_local_entropy)
print(f"(global eaverage local entropy) f3 = {f3}")

surff4 = np.mean(panorama_local_entropy_SURF) - np.mean(np.concatenate(source_local_entropy))
siftf4 = np.mean(panorama_local_entropy_SIFT) - np.mean(np.concatenate(source_local_entropy))
print(f"(differential variance of the local entropy SURF) f4 = {surff4}")
print(f"(differential variance of the local entropy SIFT) f4 = {siftf4}")

surf_f9 = abs(np.mean(std_source) - std_panorama_surf)
sift_f9 = abs(np.mean(std_source) - std_panorama_sift)
print(f"(absolute difference of standard deviations SURF) f9 = {surf_f9}")
print(f"(absolute difference of standard deviations SIFT) f9 = {sift_f9}")

f1 = open("entropies.txt", "a")
f1.write(f"\nPanorama {j}\nSURF\t\tf2 = {surff2}\tf4 = {surff4}\tf9 = {surf_f9}\n")
f1.write(f"SIFT\t\tf2 = {siftf2}\tf4 = {siftf4}\tf9 = {sift_f9}\n")
f1.write(f"f3 = {f3}\n\n")
f1.close()
