import os
import tqdm
import numpy as np
import pandas as pd
from src.multiframe.mf_parser import MFParser
from src.utils import setup_logger

MF_LABELS = [0, 1, 2]


class MFAnalyzer:
    def __init__(self, path, **kwargs):
        self.df = pd.read_csv(path, sep='\t')
        self.tracklets = MFParser(self.df).apply()
        if 'output_dir' in kwargs.keys():
            self.output_dir(kwargs['output_dir'])
        else:
            self.output_dir(os.path.dirname(path))
        self.set_logger()

    def __len__(self):
        return len(self.tracklets)

    def set_logger(self):
        path = os.path.join(self.output_dir, 'log.log')
        self.logger = setup_logger(path, name='mf_analyser')

    def output_dir(self, path):
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
                shortest_tracklet = tracklet
        return shortest_tracklet

    def get_tracklets_by_label(self, label):
        tracklets = []
        for tracklet in self.tracklets:
            if tracklet.label == label:
                tracklets.append(tracklet)
        return tracklets

    def save_tracklets(self):
        self.logger.info('Saving tracklets')
        tracklets_dir = os.path.join(self.output_dir, 'tracklets')
        os.makedirs(tracklets_dir, exist_ok=True)
        for label in MF_LABELS:
            label_dir = os.path.join(tracklets_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            tracklets = self.get_tracklets_by_label(label)
            for i, tracklet in enumerate(tracklets):
                tracklet_dir = os.path.join(label_dir, str(i))
                os.makedirs(tracklet_dir, exist_ok=True)
                tracklet_path = os.path.join(tracklet_dir, f'tracklet_uid_{tracklet.uid}.tsv')
                tracklet.save_dataframe(tracklet_path)
                tracklet.save_graphs(tracklet_dir)

    def save_tracklets_with_physical_anomalies(self):
        self.logger.info('Anomaly detection according to unphysical changes')
        tracklets_dir = os.path.join(self.output_dir, 'physical_anomalies')
        os.makedirs(tracklets_dir, exist_ok=True)
        for label in MF_LABELS:
            label_dir = os.path.join(tracklets_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            tracklets = self.get_tracklets_by_label(label)
            for i, tracklet in tqdm.tqdm(enumerate(tracklets)):
                if tracklet.physical_anomaly():
                    tracklet_dir = os.path.join(label_dir, str(i))
                    os.makedirs(tracklet_dir, exist_ok=True)
                    tracklet_path = os.path.join(tracklet_dir, f'tracklet_uid_{tracklet.uid}.tsv')
                    tracklet.save_dataframe(tracklet_path)
                    tracklet.save_graphs(tracklet_dir)
    
    def save_tracklets_with_derivatives_anomalies(self):
        self.logger.info('Anomaly detection using derivatives')
        tracklets_dir = os.path.join(self.output_dir, 'derivatives_anomalies')
        os.makedirs(tracklets_dir, exist_ok=True)
        for label in MF_LABELS:
            label_dir = os.path.join(tracklets_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            tracklets = self.get_tracklets_by_label(label)
            for i, tracklet in tqdm.tqdm(enumerate(tracklets)):
                if tracklet.derivatives_anomaly():
                    tracklet_dir = os.path.join(label_dir, str(i))
                    os.makedirs(tracklet_dir, exist_ok=True)
                    tracklet_path = os.path.join(tracklet_dir, f'tracklet_uid_{tracklet.uid}.tsv')
                    tracklet.save_dataframe(tracklet_path)
                    tracklet.save_derivatives(tracklet_dir)

