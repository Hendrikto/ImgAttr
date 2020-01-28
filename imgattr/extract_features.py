from collections import Counter

import numpy as np


def extract_band_histograms(image, bins=32):
    """Extract color histograms for each band, and merge them."""
    bands = len(image.getbands())
    histograms = np.empty(bins * bands)
    for band in range(bands):
        band_data = image.getdata(band=band)
        band_hist = np.histogram(band_data, bins=bins, range=(0, 255), density=True)[0]
        histograms[band * bins:(band + 1) * bins] = band_hist
    return histograms


def extract_color_histogram(image, bins=10):
    """Extract flattened 3D color histogram."""
    data = np.array(image.getdata())
    histogram = np.histogramdd(data, bins=bins, range=((0, 255),) * 3, density=True)[0]
    return histogram.ravel()


def extract_extrema(image):
    """Extract extrema for each bands."""
    return np.array(image.getextrema()).ravel()


def extract_palette_info(image):
    """Extract information about the color palette."""
    palette = Counter(image.getdata())
    n_colors = len(palette)
    most_frequent_colors = palette.most_common(3)
    return np.array((
        # palette size
        n_colors,
        # fractions of most frequent colors
        most_frequent_colors[0][1] / n_colors,
        most_frequent_colors[1][1] / n_colors,
        most_frequent_colors[2][1] / n_colors,
        # fraction of unique colors
        sum(1 for value in palette.values() if value == 1) / n_colors,
        # mean color frequency
        sum(palette.values()) / n_colors,
    ))


def extract_features(image):
    """Extract features from an image."""
    return np.concatenate((
        extract_band_histograms(image),
        extract_color_histogram(image),
        extract_extrema(image),
        extract_palette_info(image),
    ))
