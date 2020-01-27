from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectPercentile,
    chi2,
)
from sklearn.svm import SVC

from .evaluate_classifier import evaluate_classifier
from .extract_features import extract_features
from .prepare import (
    extract_images,
    filter_info,
    resize_images,
)
from .read_images import read_images

evaluate_classifier_parser = ArgumentParser(
    prog='evaluate_classifier',
    description='Evaluate a classifier on the dataset.',
)
evaluate_classifier_parser.add_argument(
    '--features',
    type=Path,
    default='data/features.npz',
    help='path to the CSV file containing a feature matrix',
    dest='features_path',
)
evaluate_classifier_parser.add_argument(
    '--info',
    type=Path,
    default='data/info.csv',
    help='path to the CSV file containing image information',
    dest='info_path',
)
evaluate_classifier_parser.add_argument(
    '-k', '--splits',
    type=int,
    default=10,
    help='number of splits for k-fold cross-validation',
    dest='n_splits',
)

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

select_features_parser = ArgumentParser(
    prog='select_features',
    description='Perform feature selection on the extracted features.',
)
select_features_parser.add_argument(
    '-i', '--input',
    type=Path,
    default='data/features.npz',
    help='path to input feature matrix',
    dest='input_path',
)
select_features_parser.add_argument(
    '-o', '--output',
    type=Path,
    default='data/features-selected.npz',
    help='path to output feature matrix',
    dest='output_path',
)
select_features_parser.add_argument(
    '--info',
    type=Path,
    default='data/info.csv',
    help='path to the CSV file containing image information',
    dest='info_path',
)
select_features_parser.add_argument(
    '--percentile',
    type=int,
    default=20,
    help='percentile of the features to keep',
)


class CLICommand:
    @staticmethod
    def evaluate_classifier(features_path, info_path, n_splits):
        print(f"# Loading feature matrix from '{features_path}'…")
        X = np.load(features_path)['features']
        print(f"# Reading class labels from '{info_path}'…")
        y = pd.read_csv(info_path).artist
        print(f"# Evaluating classifier using {n_splits}-fold cross-validation…")
        classifier = SVC(gamma='auto')
        metrics = evaluate_classifier(classifier, X, y, n_splits=n_splits)
        print('# Metrics:', metrics.to_string(), sep='\n')
        print('# Metrics summary:', metrics.describe().to_string(), sep='\n')

    @staticmethod
    def extract_features(info_path, input_path, output_path):
        print(f"# Reading relevant images from '{info_path}'…")
        info = pd.read_csv(info_path)
        print(f"# Extracting image features from '{input_path}'…")
        n_images = len(info)
        features = []
        for i, (image_name, image) in enumerate(read_images(input_path, names=info.new_filename)):
            print(f'{i * 100 / n_images:.2f}%\t{image_name}')
            features.append(extract_features(image))
        features = np.array(features)
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

    @staticmethod
    def select_features(input_path, output_path, info_path, percentile):
        print(f"# Loading feature matrix from '{input_path}'…")
        X = np.load(input_path)['features']
        print(f"# Reading class labels from '{info_path}'…")
        y = pd.read_csv(info_path).artist
        print(f'# Selecting most informative features…')
        feature_selector = SelectPercentile(chi2, percentile=percentile)
        X = feature_selector.fit_transform(X, y)
        print(f"# Saving selected features as '{output_path}'…")
        np.savez_compressed(output_path, features=X)


commands = {
    'evaluate_classifier': (CLICommand.evaluate_classifier, evaluate_classifier_parser),
    'extract_features': (CLICommand.extract_features, extract_features_parser),
    'extract_images': (CLICommand.extract_images, extract_images_parser),
    'filter_info': (CLICommand.filter_info, filter_info_parser),
    'resize_images': (CLICommand.resize_images, resize_images_parser),
    'select_features': (CLICommand.select_features, select_features_parser),
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
