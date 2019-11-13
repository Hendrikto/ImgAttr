import os.path
from zipfile import ZipFile

import pandas as pd

max_image_bytes = 1024 ** 2


def frequent_artists(info, threshold=100):
    """Find artists with a minimum number of images."""
    artist_counts = info.artist.value_counts()
    return artist_counts[artist_counts >= threshold].keys()


def filter_info(csv_path):
    info = pd.read_csv(csv_path)

    # filter out images which are too big
    info = info[info.size_bytes <= max_image_bytes]

    # filter out artists with too few images
    info = info[info.artist.isin(frequent_artists(info))]

    # select relevant columns
    info = info[['artist', 'new_filename']]

    # reset the index
    return info.reset_index(drop=True)


def extract_images(input_path, output_path, names):
    """Selectively extract relevant images from a zip archive."""
    names = set(names)
    rejected = set()
    with ZipFile(input_path) as input_zip, \
            ZipFile(output_path, 'a') as output_zip:
        for image_info in input_zip.infolist():
            image_name = os.path.basename(image_info.filename)
            if image_name not in names:
                continue

            if image_info.file_size > max_image_bytes:
                rejected.add(image_name)
            else:
                output_zip.writestr(image_name, input_zip.read(image_info.filename))
    return rejected
