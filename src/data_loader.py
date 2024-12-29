import polars as pl
import pathlib
import numpy as np
from tensorflow.keras.utils import img_to_array, load_img
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import filetype
from scaler import Scaler


class DataLoader:
    data_path: pathlib.Path
    df: pl.DataFrame


    def __init__(self, data_path: str):
        self.data_path = pathlib.Path(data_path)
        self.df = pl.read_csv(self.data_path / "metadata.csv")


    def dataset(self, target_size=(224, 224), batch_size=32, scaling=True) -> tf.data.Dataset:
        """
        Gets TensorFlow Dataset to train model.
        This loads the data in batches, allowing for a large dataset
        (i.e. one that does not fit in memory) to be used.

        Args:
            target_size : tuple[int, int],  default=(224,224)
                Size to resize images to (height, width).

            batch_size : int, default=32
                Size of the batch of training data.

            scaling : bool, default=True
                Whether or not to scale coordinates.
                Uses min-max scaling.
        Returns:
            tf.data.Dataset:
                Dataset to be used in training.
        """
        # Clean-up dataset
        self.df = self.__filter_out_gifs(self.df)
        self.df = self.__filter_out_non_existent_files(self.df)

        paths: pl.Series = str(self.data_path) + "/" + self.df["year"].cast(pl.String) 
        paths += "/" + self.df["id"].cast(pl.String) + ".jpg"
        paths = paths.to_numpy()
        
        if scaling:
            years = Scaler.scale_years(self.df["year"].to_numpy())
            lats = Scaler.scale_latitudes(self.df["latitude"].to_numpy())
            lons = Scaler.scale_longitudes(self.df["longitude"].to_numpy())
        else:
            years = self.df["year"].to_numpy()
            lats = self.df["latitude"].to_numpy()
            lons = self.df["longitude"].to_numpy()

        dataset = tf.data.Dataset.from_tensor_slices((paths, years, lats, lons))
        dataset = dataset.map(
            lambda path, year, lat, lon: DataLoader.load_image_and_labels(path, year, lat, lon, image_shape=target_size)
        )

        dataset = dataset.shuffle(buffer_size=len(paths)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset


    def consolidate_metadata(self):
        metadata_df = self.__load_all_metadata(self.data_path)
        metadata_df.write_csv(self.data_path / "metadata.csv")


    @staticmethod
    def load_image_and_labels(path: str, year: int, lat: float, lon: float, image_shape=(224, 224)):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_shape) / 255.0
        return img, {
            "year": year,
            "lat": lat,
            "lon": lon
        }


    def __filter_out_gifs(self, df: pl.DataFrame) -> pl.DataFrame:
        def is_gif(filepath: str) -> bool:
            guess = filetype.guess(filepath)
            return guess.mime == "image/gif"
        
        original_columns = df.columns
        df = df.with_columns(
            (
                str(self.data_path) + "/" + pl.col("year").cast(pl.String) + "/" + pl.col("id").cast(pl.String) + ".jpg"
            ).alias("filepath")
        )
        df = df.with_columns(
            (
                df["filepath"].map_elements(is_gif, return_dtype=pl.Boolean)
            ).alias("is_gif")
        )
        df = df.filter(~pl.col("is_gif"))
        return df.select(original_columns)


    def __filter_out_non_existent_files(self, df: pl.DataFrame) -> pl.DataFrame:
        def file_exists(filepath: str) -> bool:
            file = pathlib.Path(filepath)
            return file.is_file()
        
        original_columns = df.columns
        df = df.with_columns(
            (
                str(self.data_path) + "/" + pl.col("year").cast(pl.String) + "/" + pl.col("id").cast(pl.String) + ".jpg"
            ).alias("filepath")
        )
        df = df.with_columns(
            (
                df["filepath"].map_elements(file_exists, return_dtype=pl.Boolean)
            ).alias("file_exists")
        )
        df = df.filter(pl.col("file_exists"))
        return df.select(original_columns)


    def __load_all_metadata(self, images_dir: pathlib.Path) -> pl.DataFrame:
        """
        Loads all metadata files into a single DataFrame.
        """
        csvs = []
        for d in images_dir.iterdir():
            if d.is_dir():
                df = pl.read_csv(d / "metadata.csv")
                csvs.append(df)
        return pl.concat(csvs)


def load_all_data(dir: pathlib.Path, target_size=(224, 224),
    scaling=True, scaler=MinMaxScaler(),
    start_year=1900, end_year=2025
) -> tuple[np.ndarray, np.ndarray, ]:
    """
    Loads images and labels from a directory.

    Args:
        dir : pathlib.Path
            Directory year subdirectories.

        target_size : tuple[int, int],  default=(224,224)
            Size to resize images to (height, width).

        scaling : bool, default=True
            Whether or not to scale coordinates.

        scaler : Optional[sklearn.StandardScaler], default=MinMaxScaler
            The scaler chosen for scaling. Will be returned by the function to allow for easy descaling.

        start_year : int, default=1900
            First year of the range to include images from (inclusive).

        end_year : int, default=2025
            Last year of the range to include images from (exclusive).

    Returns:
        np.ndarray
            Array of preprocessed images

        np.ndarray
            Array of labels (year, latitude, longitude)

        scaler
            The scaler used.
    """
    if scaling and not scaler:
        raise ValueError("A scaler must be provided if scaling was chosen.")

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
        label_arr[:, 1:] = scaler.fit_transform(label_arr[:, 1:])

    return image_arr, label_arr, scaler


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

