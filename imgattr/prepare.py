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
