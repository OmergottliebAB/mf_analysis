import os
import numpy as np
from visualisation import plot_tracklet_position, plot_kinematics


class Tracklet:
    def __init__(self, df):
        self.df = df
        self.init()

    def init(self):
        self.label = self.df.loc[0, 'label']
        self.sub_class = self.df.loc[0, 'sub_class']
        self.age = self.df['age'].iloc[-1]
        self.uid = self.df.loc[0, 'uid']
        self.frames = self.df['name'].to_numpy()
        self.scores = self.df['score'].to_numpy()
        # Kalman tracked parameters
        self.world_width = self.df['world_width'].to_numpy()
        self.world_height = self.df['world_height'].to_numpy()
        self.lat_dist = self.df['lat_dist'].to_numpy()
        self.long_dist = self.df['long_dist'].to_numpy()
        self.abs_vel_x = self.df['abs_vel_x'].to_numpy()
        self.abs_vel_z = self.df['abs_vel_z'].to_numpy()
        self.abs_acc_x = self.df['abs_acc_x'].to_numpy()
        self.abs_acc_z = self.df['abs_acc_z'].to_numpy()
        self.rel_vel_x = self.df['rel_vel_x'].to_numpy()
        self.rel_vel_z = self.df['rel_vel_z'].to_numpy()
        self.rel_acc_x = self.df['rel_acc_x'].to_numpy()
        self.rel_acc_z = self.df['rel_acc_z'].to_numpy()
        self.orientation = self.df['orientation'].to_numpy()
        # Derived variables from different logics
        self.lane_associaation = self.df['lane_association'].to_numpy()
        self.is_cipv = self.df['is_cipv'].to_numpy()
        self.is_occluded()
        # image plane parameters
        self.x_center = self.df['x_center'].to_numpy()
        self.y_center = self.df['y_center'].to_numpy()
        self.width = self.df['width'].to_numpy()
        self.height = self.df['height'].to_numpy()
        self.d3_separation = self.df['d3_separation'].to_numpy()
        self.sf_confirmed = self.df['sf_confirmed'].to_numpy()
        self.bbox_aspect_ratio()

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        tracklet_path = os.path.join(path, f'tracklet.tsv')
        self.save_dataframe(tracklet_path)
        self.save_graphs(path)

    def save_graphs(self, path):
        plot_tracklet_position(self.lat_dist, self.long_dist, path)
        self._plot_kinematics(path)

    def _plot_kinematics(self, path):
        axis_dict = {'x_axis': {'lat_dist':  {'vector': self.lat_dist, 'units': 'm'},
                                'abs_vel_x': {'vector': self.abs_vel_x, 'units': 'm/s'},
                                'abs_acc_x': {'vector': self.abs_acc_x, 'units': 'm/s^2'},
                                'rel_vel_x': {'vector': self.rel_vel_x, 'units': 'm/s'},
                                'rel_acc_x': {'vector': self.rel_acc_x, 'units': 'm/s^2'}},
                     'z_axis': {'long_dist': {'vector': self.long_dist, 'units': 'm'},
                                'abs_vel_z': {'vector': self.abs_vel_z, 'units': 'm/s'},
                                'abs_acc_z': {'vector': self.abs_acc_z, 'units': 'm/s^2'},
                                'rel_vel_z': {'vector': self.rel_vel_z, 'units': 'm/s'},
                                'rel_acc_z': {'vector': self.rel_acc_z, 'units': 'm/s^2'}}}
        x = self.df['age'].to_numpy()
        file_path = os.path.join(path, 'kinematics.png')
        plot_kinematics(x, axis_dict, file_path)

    def save_dataframe(self, path):
        self.df = self.df.drop(['index'], axis=1)
        self.df.to_csv(path, sep='\t', index=False)

    def bbox_aspect_ratio(self):
        self.aspect_ratio = self.width / self.height

    def is_occluded(self):
        if 'is_occluded' in self.df.columns:
            self.is_occluded = self.df['is_occluded'].to_numpy()
        else:
            self.is_occluded = np.full(len(self.df), np.nan)

    def longitudinal_velocity_sign_change(self):
        flag = False
        for i in range(len(self.abs_vel_z)-1):
            curr_vel = self.abs_vel_z[i]
            next_vel = self.abs_vel_z[i+1]
            if self._sign_difference(curr_vel, next_vel) and abs(next_vel - curr_vel) > 1:
                flag = True
        return flag

    @staticmethod
    def _sign_difference(x, y):
        if (x < 0 and y >= 0) or (x >= 0 and y < 0):
            return True
        else:
            return False

