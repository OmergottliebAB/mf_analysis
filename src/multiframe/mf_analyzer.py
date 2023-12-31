import os
import tqdm
import numpy as np
import pandas as pd
from mf_parser import MFParser
from src.utils import setup_logger

MF_LABELS = [0, 1, 2]

logger = setup_logger(name='mf_analyser')


class MFAnalyzer:
    def __init__(self, path):
        self.df = pd.read_csv(path, sep='\t')
        self.tracklets = MFParser(self.df).apply()
        self.output_dir(path)

    def __len__(self):
        return len(self.tracklets)

    def output_dir(self, path):
        path = os.path.dirname(path)
        self.output_dir = os.path.join(path, 'output')
        os.makedirs(self.output_dir, exist_ok=True)

    def get_labels(self):
        return np.unique(self.df['label'].to_numpy())

    def longest_tracklet_by_label(self, label):
        tracklets = self.get_tracklets_by_label(label)
        age = 0
        for tracklet in tracklets:
            if tracklet.age > age:
                age = tracklet.age
                longest_tracklet = tracklet
        return longest_tracklet

    def shortes_tracklet_by_label(self, label):
        tracklets = self.get_tracklets_by_label(label)
        age = 10e6
        for tracklet in tracklets:
            if tracklet.age < age:
                age = tracklet.age
                longest_tracklet = tracklet
        return longest_tracklet

    def get_tracklets_by_label(self, label):
        tracklets = []
        for tracklet in self.tracklets:
            if tracklet.label == label:
                tracklets.append(tracklet)
        return tracklets

    def save_tracklets(self):
        tracklets_dir = os.path.join(self.output_dir, 'tracklets')
        os.makedirs(tracklets_dir, exist_ok=True)
        for label in MF_LABELS:
            label_dir = os.path.join(tracklets_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            tracklets = self.get_tracklets_by_label(label)
            for i, tracklet in enumerate(tracklets):
                tracklet_dir = os.path.join(label_dir, str(i))
                tracklet.save(tracklet_dir)

    def save_physical_anomalies(self):
        tracklets_dir = os.path.join(self.output_dir, 'physical_anomalies')
        os.makedirs(tracklets_dir, exist_ok=True)
        for label in MF_LABELS:
            label_dir = os.path.join(tracklets_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            tracklets = self.get_tracklets_by_label(label)
            for i, tracklet in enumerate(tracklets):
                if tracklet.longitudinal_velocity_sign_change():
                    tracklet_dir = os.path.join(label_dir, str(i))
                    tracklet.save(tracklet_dir)


if __name__ == "__main__":
    path = '/home/omer/B2B/multiframe/unsupervised_analysis/stanch_las_cruces_10_fps__test/cametra_interface_output.tsv'
    mfa = MFAnalyzer(path)
    mfa.save_physical_anomalies()
    print('')
