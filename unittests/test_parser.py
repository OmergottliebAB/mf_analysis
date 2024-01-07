import unittest
import numpy as np
import pandas as pd
from src.multiframe.mf_parser import MFParser

PATH = 'input_files/cametra_interface_output.tsv'
class TestParser(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv(PATH, sep='\t')
        self.num_frames = len(np.unique(self.df['name'].to_numpy()))
        self.tracklets = MFParser(self.df).apply()

    def test_tracklet_age(self):
        for tracklet in self.tracklets:
            self.assertGreaterEqual(self.num_frames, tracklet.age)

    def test_empty_tracklet(self):
        for tracklet in self.tracklets:
            self.assertLess(0, len(tracklet.df))




if __name__ == '__main__':
    unittest.main()