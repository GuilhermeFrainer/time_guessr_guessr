import polars as pl
import os
import pathlib
import numpy as np
from tensorflow.keras.utils import img_to_array, load_img
from sklearn.preprocessing import MinMaxScaler


def load_all_data(dir: pathlib.Path, target_size=(224, 224), scaling=True, start_year=1900, end_year=2025) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads images and labels from a directory.

    Args:
        dir : pathlib.Path
            Directory year subdirectories.

        target_size : tuple[int, int],  default=(224,224)
            Size to resize images to (height, width).

        scaling : bool, default=True
            Whether or not to scale coordinates.

        start_year : int, default=1900
            First year of the range to include images from (inclusive).

        end_year : int, default=2025
            Last year of the range to include images from (exclusive).

    Returns:
        np.ndarray
            array of preprocessed images

        np.ndarray
            array of labels (year, latitude, longitude)
    """
    if isinstance(dir, str):
        dir = pathlib.Path(dir)

    images = []
    labels = []
    for d in sorted(dir.iterdir()):
        if int(d.name) < start_year:
            continue
        elif int(d.name) >= end_year:
            break
        new_images, new_labels = __load_data_for_year(d, target_size=target_size)
        images.append(new_images)
        labels.append(new_labels)
    image_arr = np.concatenate(images)
    label_arr = np.concatenate(labels)

    if scaling:
        scaler = MinMaxScaler()
        label_arr[:, 1:] = scaler.fit_transform(label_arr[:, 1:])

    return image_arr, label_arr


def __load_data_for_year(dir: pathlib.Path, target_size=(224,224)) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads images and labels for a year.

    Args:
        dir : pathlib.Path
            Directory containing images and a metadata.csv file.

        target_size : tuple[int, int],  default=(224,224)
            Size to resize images to (height, width).

    Returns:
        np.ndarray
            Array of preprocessed images.

        np.ndarray
            Array of labels (year, latitude, longitude).
    """
    if isinstance(dir, str):
        dir = pathlib.Path(dir)
    
    df = pl.read_csv(dir / "metadata.csv")

    images = []
    labels = []
    for id in df["id"]:
        filename = str(id) + ".jpg"
        try:
            img = load_img(dir / filename, target_size=target_size)
            arr = img_to_array(img) / 255.0
            images.append(arr)

            # Saves metadata of file
            metadata = df.filter(pl.col("id") == id).select(["year", "latitude", "longitude"])
            labels.append(metadata.to_numpy()[0])
        except FileNotFoundError:
            continue

    return np.array(images), np.array(labels)

