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
