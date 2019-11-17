from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from .extract_features import extract_features
from .prepare import (
    extract_images,
    filter_info,
    resize_images,
)
from .read_images import read_images

extract_features_parser = ArgumentParser(
    prog='extract_features',
    description='Generate features from the images.',
)
extract_features_parser.add_argument(
    '-i', '--input',
    type=Path,
    default='data/images-resized.zip',
    help='path to the zip file with the input images',
    dest='input_path',
)
extract_features_parser.add_argument(
    '-o', '--output',
    type=Path,
    default='data/features.npz',
    help='path to the output file',
    dest='output_path',
)
extract_features_parser.add_argument(
    '--info',
    type=Path,
    default='data/info.csv',
    help='path to the CSV file with the relevant images',
    dest='info_path',
)

extract_images_parser = ArgumentParser(
    prog='extract_images',
    description='Extract relevant files from a zip archive.',
)
extract_images_parser.add_argument(
    '-i', '--input',
    type=Path,
    required=True,
    help='path to the input zip file',
    dest='input_path',
)
extract_images_parser.add_argument(
    '-o', '--output',
    type=Path,
    default='data/images.zip',
    help='path to the output zip file',
    dest='output_path',
)
extract_images_parser.add_argument(
    '--info',
    type=Path,
    default='data/info.csv',
    help='path to the CSV file with the relevant images',
    dest='info_path',
)

filter_info_parser = ArgumentParser(
    prog='filter_info',
    description='Filter the information given as a CSV file.',
)
filter_info_parser.add_argument(
    '-i', '--input',
    type=str,
    required=True,
    help='path to the raw CSV file',
    dest='input_path',
)
filter_info_parser.add_argument(
    '-o', '--output',
    type=str,
    default='data/info.csv',
    help='path to the output CSV file',
    dest='output_path',
)

resize_images_parser = ArgumentParser(
    prog='resize_images',
    description='Resize images to fit into a maximum size.',
)
resize_images_parser.add_argument(
    '-i', '--input',
    type=Path,
    default='data/images.zip',
    help='path to the input zip file',
    dest='input_path',
)
resize_images_parser.add_argument(
    '-o', '--output',
    type=Path,
    default='data/images-resized.zip',
    help='path to the output zip file',
    dest='output_path',
)


class CLICommand:
    @staticmethod
    def extract_features(info_path, input_path, output_path):
        print(f"# Reading relevant images from '{info_path}'…")
        info = pd.read_csv(info_path)
        print(f"# Extracting image features from '{input_path}'…")
        features = np.vstack([
            extract_features(image)
            for _, image in read_images(input_path, names=info.new_filename)
        ])
        print(f"# Saving features as '{output_path}'…")
        np.savez_compressed(output_path, features=features)

    @staticmethod
    def extract_images(info_path, input_path, output_path):
        print(f"# Reading relevant images from '{info_path}'…")
        info = pd.read_csv(info_path)
        print(f"# Extracting relevant images from '{input_path}' into '{output_path}'…")
        rejected = extract_images(input_path, output_path, info.new_filename)
        print(f"# Removing rejected images from '{info_path}'…")
        info = info[~info.new_filename.isin(rejected)]
        info.to_csv(info_path, index=False)

    @staticmethod
    def filter_info(input_path, output_path):
        print(f'# Filtering info {input_path!r}…')
        info = filter_info(input_path)
        print(f'# Saving filtered info as {output_path!r}…')
        info.to_csv(output_path, index=False)

    @staticmethod
    def resize_images(input_path, output_path):
        print(f"# Resizing images from '{input_path}' into '{output_path}'…")
        resize_images(input_path, output_path)


commands = {
    'extract_features': (CLICommand.extract_features, extract_features_parser),
    'extract_images': (CLICommand.extract_images, extract_images_parser),
    'filter_info': (CLICommand.filter_info, filter_info_parser),
    'resize_images': (CLICommand.resize_images, resize_images_parser),
}

command_parser = ArgumentParser(
    prog='python -m imgattr',
    description='Attribute image authorship, based on image features.',
)
command_parser.add_argument(
    'command',
    type=str,
    choices=commands,
    help='command to execute',
)
