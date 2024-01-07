import tqdm
import logging
import numpy as np
from itertools import chain
from src.multiframe.tracklet import Tracklet

MF_LABELS = [0, 1, 2]

logger: logging.Logger = logging.getLogger("mf_analyser")


class MFParser:
    def __init__(self, df):
        self.df = df

    def apply(self):
        tracklets = []
        logger.info('Parsing MF dataframe')
        for label in MF_LABELS:
            uids = np.unique(
                self.df[self.df['label'] == label]['uid'].to_numpy())
            for uid in tqdm.tqdm(uids):
                uid_tracklets = self._get_tracklets(uid, label)
                tracklets.append(uid_tracklets)
        return list(chain.from_iterable(tracklets))

    def _get_tracklets(self, uid: int, label: int):
        tracklets = []
        uid_df = self.df[(self.df['uid'] == uid) & (
            self.df['label'] == label)].reset_index()
        # Assumption: a new tracklet start when diff in age vector is negative
        diff = np.diff(uid_df['age'].to_numpy())
        # condition for singel tracklet with given uid
        if self.all_positive(diff):
            return [Tracklet(uid_df)]

        indeces = [i for i, x in enumerate(diff) if x < 0]
        # condition for two tracklets with given uid
        if len(indeces) == 1:
            indeces.append(len(uid_df)-2)
        i_prev = 0
        for i_curr in indeces:
            df_tmp = uid_df[i_prev:i_curr+1].reset_index()
            if not df_tmp.empty:
                tracklets.append(Tracklet(df_tmp))
            i_prev = i_curr+1
        return tracklets

    @staticmethod
    def all_positive(vector):
        return all(x > 0 for x in vector)

    @staticmethod
    def is_monotonic(vector: np.array):
        return all(vector[i] <= vector[i + 1] for i in range(len(vector) - 1))
