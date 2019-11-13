def frequent_artists(info, threshold=100):
    """Find artists with a minimum number of images."""
    artist_counts = info.artist.value_counts()
    return artist_counts[artist_counts >= threshold].keys()
