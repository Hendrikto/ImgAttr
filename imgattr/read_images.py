from zipfile import ZipFile

from PIL import Image


def read_images(zip_path, names=None):
    with ZipFile(zip_path) as zip_file:
        for name in zip_file.namelist() if names is None else names:
            with zip_file.open(name) as image_file:
                image = Image.open(image_file)
                image.load()
            yield name, image
