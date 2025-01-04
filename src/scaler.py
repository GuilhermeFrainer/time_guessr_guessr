import numpy as np


class Scaler:
    YEAR_MIN = 1900
    YEAR_MAX = 2024

    LAT_MIN = -90
    LAT_MAX = 90

    LON_MIN = -180
    LON_MAX = 180


    @staticmethod
    def scale_labels(labels: np.ndarray) -> np.ndarray:
        """
        Scales label data.

        Args:
            labels : np.ndarray
                Labels to be scaled.
                Should be an n-dimensional array of 3-dimensional arrays with [year, latitude, longitude].

        Returns:
            np.ndarray:
                Scaled labels
        """
        out_labels = labels
        out_labels[:, 0:1] = Scaler.scale_years(labels[:, 0:1])
        out_labels[:, 1:2] = Scaler.scale_latitudes(labels[:, 1:2])
        out_labels[:, 2:3] = Scaler.scale_longitudes(labels[:, 2:3])
        return out_labels


    @staticmethod
    def scale_longitudes(lons: np.ndarray) -> np.ndarray:
        return Scaler.__min_max_scaling(lons, min=Scaler.LON_MIN, max=Scaler.LON_MAX)


    @staticmethod
    def scale_latitudes(lats: np.ndarray) -> np.ndarray:
        return Scaler.__min_max_scaling(lats, min=Scaler.LAT_MIN, max=Scaler.LAT_MAX)


    @staticmethod
    def scale_years(years: np.ndarray) -> np.ndarray:
        return Scaler.__min_max_scaling(years, min=Scaler.YEAR_MIN, max=Scaler.YEAR_MAX)


    @staticmethod
    def descale_longitudes(lons: np.ndarray) -> np.ndarray:
        return Scaler.__min_max_descaling(lons, min=Scaler.LON_MIN, max=Scaler.LON_MAX)
    

    @staticmethod
    def descale_latitudes(lats: np.ndarray) -> np.ndarray:
        return Scaler.__min_max_descaling(lats, min=Scaler.LAT_MIN, max=Scaler.LAT_MAX)
    

    @staticmethod
    def descale_years(years: np.ndarray) -> np.ndarray:
        return Scaler.__min_max_descaling(years, min=Scaler.YEAR_MIN, max=Scaler.YEAR_MAX)
    

    @staticmethod
    def descale_preds(preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        year = Scaler.descale_years(preds["year"])
        lat = Scaler.descale_latitudes(preds["lat"])
        lon = Scaler.descale_longitudes(preds["lon"])
        return {"year": year, "lat": lat, "lon": lon}


    @staticmethod
    def __min_max_scaling(data: np.ndarray, min=0, max=1) -> np.ndarray:
        """
        Generic function to scale data in a known range to [0, 1].
        Scaling is performed as such:

        y_scaled = \\frac{y - y_min}{y_max - y_min}

        Args:
            data : np.ndarray
                Data to be scaled.

            min : int, default=0
                Smallest element in data range.

            max : int, default=1
                Largest element in data range.

        Returns:
            np.ndarray:
                Scaled data.
        """
        return (data - min) / (max - min)
    

    @staticmethod
    def __min_max_descaling(scaled_data: np.ndarray, min=0, max=1) -> np.ndarray:
        return scaled_data * (max - min) + min

