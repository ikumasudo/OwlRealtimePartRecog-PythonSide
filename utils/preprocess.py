from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import tsfresh
from tsfresh.feature_extraction import settings
from tsfresh.utilities.dataframe_functions import impute
from typing import Tuple, Optional


class Preprocess:
    def __init__(self, feature_path: str, n_features: int):
        importance = pd.read_csv(feature_path)
        self.feature_names = importance["name"][:n_features]


    def extract_peaks(self, data2d: np.ndarray, r: int = 10, dist_mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, int]:
        # data2d: (n_sensors, n_timepoints)
        # r: ピークを中心にして，左右に抽出する範囲
        # dist_mult: ピーク同士の感覚をr*dist_multだけ空ける

        # data2dからピークを取り出す
        datalen = data2d.shape[1]               # データの長さ
        std_sensors = data2d.std(axis=1)        # 各センサーのSTD
        dominant_sensor = std_sensors.argmax()  # もっともSTDが大きいセンサーのidx
        # print("dominant sensor:", dominant_sensor)

        peak_index, _ = find_peaks(
            data2d[dominant_sensor, :],
            distance=r*dist_mult,
            prominence=0.2
            )

        adopted_peak_index = []
        clipped_data = []
        for i, a_peak_index in enumerate(peak_index):
            left_end = a_peak_index - r
            right_end = a_peak_index + r

            if (left_end < 0) or (right_end >= datalen):
                continue

            clipped = data2d[:, left_end:right_end]
            clipped_data.append(clipped)
            adopted_peak_index.append(a_peak_index)

        if not adopted_peak_index:
            return None, None, None
        else:
            # (n_peaks,), (n_peaks, n_sensors, n_timepoints), int
            return np.stack(adopted_peak_index), np.stack(clipped_data), dominant_sensor


    def to_tsfresh_format(self, peaks: np.ndarray) -> np.ndarray:
        # peaks: (n_peaks, n_sensors, n_timepoints)
        length = peaks.shape[2]
        arr = []
        for id, peak in enumerate(peaks):
            id_col = np.array([[id]*length]).transpose()
            time_col = np.arange(length).reshape(-1, 1)
            data_col = peak.transpose()
            stacked = np.hstack([id_col, time_col, data_col])
            arr.append(stacked)

        timeseries = pd.DataFrame(
            np.vstack(arr), 
            columns=[
                "id", "time", "sensor0", "sensor1", "sensor2", "sensor3"
                ]
            )
        timeseries = timeseries.astype({"id": "int64", "time": "int64"})
        return timeseries


    def tsfresh_features(self, data: np.ndarray) -> pd.DataFrame:
        # data: tsfresh format
        top_n_features = tsfresh.feature_extraction.settings.from_columns(self.feature_names)
        features = tsfresh.extract_features(
            data,
            column_id="id",
            column_sort="time",
            n_jobs=0,       # n_jobs=0でかなり高速化できる
            disable_progressbar=True,
            kind_to_fc_parameters=top_n_features
            )
        
        impute(features)
        
        # 列並べ替え
        features = features[self.feature_names]
        return features


    def __call__(self, data: np.ndarray) -> Optional[np.ndarray]:
        # data: (n_sensors, n_timepoints)
        _, peaks, _ = self.extract_peaks(data)
        if peaks is not None:
            peaks_tsf = self.to_tsfresh_format(peaks)
            features = self.tsfresh_features(peaks_tsf)
            return features
        else:
            return None
        
    def get_peaks(self, data: np.ndarray) -> Optional[np.ndarray]:
        _, peaks, _ = self.extract_peaks(data)
        return peaks


if __name__ == "__main__":
    import time

    preprocess = Preprocess(".\data\OwlNotebook22-FeatureImportance.csv", 300)
    start = time.time()
    data = pd.read_csv("./data/sample.csv", header=None).values.transpose()[1:]
    data = preprocess(data)
    print(data.shape, type(data))
    print(time.time() - start)
    
    # print(preprocess.get_peaks(data))
    
