import ants
import numpy as np
import nibabel as nib
from functions import get_npy_paths

from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.distance import cdist


def load_histograms(root_directory):
    paths = get_npy_paths(root_directory)

    with ProcessPoolExecutor() as executor:
        histograms = {modality: list(executor.map(np.load, histograms)) for modality, histograms in paths.items()}
        # histograms = [histogram for histogram in executor.map(np.load, paths)]

    return histograms


def calculate_histogram(img_path, bins=100):
    img = nib.load(img_path).get_fdata()
    hist, _ = np.histogram(img, bins=bins)
    return hist  # / np.sum(hist)


def cluster_histograms(histograms, num_clusters):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    kmeans.fit(histograms)
    return kmeans


def find_closest_ref_histogram(img_histogram, ref_histograms, kmeans):
    cluster = kmeans.predict([img_histogram])[0]
    distances = cdist([kmeans.cluster_centers_[cluster]], ref_histograms)
    closest_hist_idx = np.argmin(distances)
    return closest_hist_idx


def match_histogram(img_path, ref_histograms, kmeans, ref_img_paths, verbose=False):
    img = ants.image_read(img_path)
    img_hist = calculate_histogram(img_path)
    closest_hist_idx = find_closest_ref_histogram(img_hist, ref_histograms, kmeans)
    ref_img_path = ref_img_paths[closest_hist_idx]

    if verbose:
        print(f'Matching histogram for: {img_path.split("/")[-1]} ({ref_img_path.split("/")[-1]})')

    ref_img = ants.image_read(ref_img_path)
    matched_img = ants.histogram_match_image(img, ref_img)
    return matched_img
