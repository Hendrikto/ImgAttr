# ImgAttr

Attribute image authorship, based on image features.

## Preparing the Dataset

Because of its large size, the dataset is not included in this repository. It must be downloaded from the [Painter by Number](https://www.kaggle.com/c/painter-by-numbers/data) Kaggle competition. The following files are required:

* `train.zip`
* `test.zip`
* `replacements_for_corrupted_files.zip`
* `all_data_info.csv`

Once you have downloaded the required files, you can prepare the dataset. First, filter the information CSV file, to select all suitable images:

```bash
~/project $ python -m imgattr filter_info -i /path/to/all_data_info.csv
```

Next, extract the relevant train and test images from the zip archives:

```bash
~/project $ python -m imgattr extract_images -i /path/to/train.zip
~/project $ python -m imgattr extract_images -i /path/to/test.zip
```

Some of the images in the dataset were corrupted, but there are replacements for these in the `replacements_for_corrupted_files.zip` file. Overwrite the corrupted files in the extracted images zip file.

```bash
~/project/data $ unzip /path/to/replacements_for_corrupted_files.zip
~/project/data $ zip /path/to/images.zip *.jpg
```

During the filtering and extraction steps, we already rejected images which are too big (size in bytes). This is done because the Pillow library which is used cannot handle some of the excessively large files, and processing would take too long, since a few of the images from the dataset were in excess of 100MB. The reason this takes place in two steps, is that the `size_bytes` column in `all_data_info.csv` contains incorrect sizes for some of the files.

As the last step of the preprocessing pipeline, images which are too large (size in pixels) are resized to fit into a 500 by 500 square:

```bash
~/project/ $ python -m imgattr resize_images
```
