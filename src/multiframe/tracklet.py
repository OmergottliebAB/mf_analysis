import os
import logging
import numpy as np
from src.multiframe.smoothness import second_derivative_anomaly
from src.multiframe.visualisation import plot_tracklet_position, plot_kinematics, plot_bbox_params, plot_derivatives

logger: logging.Logger = logging.getLogger("mf_analyser")


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

    def save_graphs(self, path):
        plot_tracklet_position(self.lat_dist, self.long_dist, path)
        self._plot_kinematics(path)
        self._plot_bbox_parameters(path)

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

    def _plot_bbox_parameters(self, path):
        params_dict = {'world_width': {'vector': self.world_width, 'units': 'm'},
                       'world_height': {'vector': self.world_height, 'units': 'm'}}
        x = self.df['age'].to_numpy()
        file_path = os.path.join(path, 'bbox_params.png')
        plot_bbox_params(x, params_dict, file_path)

    def save_dataframe(self, path):
        if 'index' in self.df.columns:
            self.df = self.df.drop(['index'], axis=1)
        self.df.to_csv(path, sep='\t', index=False)

    def bbox_aspect_ratio(self):
        self.aspect_ratio = self.width / self.height

    def is_occluded(self):
        if 'is_occluded' in self.df.columns:
            self.is_occluded = self.df['is_occluded'].to_numpy()
        else:
            self.is_occluded = np.full(len(self.df), np.nan)

    def physical_anomaly(self):
        # TODO: consider truncation boxes
        check_functions = {
            0: self.world_height_anomaly,
            2: self.longitudinal_velocity_sign_change
        }

        check_func = check_functions.get(self.label)

        return len(self.df) >= 5 and (check_func() if check_func else False)

    def longitudinal_velocity_sign_change(self):
        flag = False
        idx = max(int(len(self.df) * 0.1),
                  (self.df['age'] - 10).abs().idxmin())
        for i in range(idx, len(self.abs_vel_z)-1):
            curr_vel = self.abs_vel_z[i]
            next_vel = self.abs_vel_z[i+1]
            if self._sign_difference(curr_vel, next_vel) and abs(next_vel - curr_vel) > 2:
                flag = True
                logger.info(
                    f'Longitudinal velocity anomaly at frame: {self.frames[i+1]} for label:{self.label} ; uid:{self.uid}')
        return flag

    def world_height_anomaly(self):
        idx = int(len(self.df) * 0.1)
        x = 2*np.std(self.world_height[idx:]) + \
            np.mean(self.world_height[idx:])
        flag = np.any(self.world_height[idx:] > min(x, 2.2))
        if flag:
            logger.info(
                f'Height anomaly for label:{self.label} ; uid:{self.uid}')
        return flag

    @staticmethod
    def _sign_difference(x, y):
        if (x < 0 and y >= 0) or (x >= 0 and y < 0):
            return True
        else:
            return False

    def derivatives_anomaly(self):
        # TODO: test axis cross correlation
        # TODO: test second derivative auto correlation
        # TODO: consider truncation boxes
        # Apply anomaly detection on object with enough history for Kalman filter to converge
        if self.age < 10:
            return False
        vector_dict = {'lat_dist': self.lat_dist, 'long_dist': self.long_dist, 'lat_vel': self.abs_vel_x, 'longi_vel': self.abs_vel_z}
        self.vectors_for_plot = {}
        flags = []
        for key,x in vector_dict.items():
            flag, indices = second_derivative_anomaly(x, self.df['age'].to_numpy())
            flags.append(flag)
            if flag:
                self.vectors_for_plot[key] = x
                logger.info(
                    f'Second derivative anomaly at {key} for label:{self.label} with uid:{self.uid} at frame:{self.frames[indices]}')
        return any(flags)

    def save_derivatives(self, path):
        os.makedirs(path, exist_ok=True)
        for key, value in self.vectors_for_plot.items():
            x = self.df['age'].to_numpy()
            file_path = os.path.join(path, f'{key}_derivatives.png')
            plot_derivatives(x, value, key, file_path)
