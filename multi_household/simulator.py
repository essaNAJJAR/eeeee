import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.data_loader import generate_smart_meter_data, APPLIANCES_UKDALE, APPLIANCE_PROFILES_UKDALE
from preprocessing.preprocessor import preprocess_signal
from feature_extraction.vmd import feedback_vmd
from feature_extraction.teo import teager_energy_operator


class MultiHouseholdSimulator:
    def __init__(self, num_households=20, duration=86400, sampling_rate=1/6):
        self.num_households = num_households
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.household_data = {}
        self.grid_data = None
        self.results = {}

    def generate_data(self, num_households=None, duration=None):
        if num_households:
            self.num_households = num_households
        if duration:
            self.duration = duration

        total_samples = int(self.duration / self.sampling_rate)
        self.grid_data = np.zeros(total_samples)

        for h in range(self.num_households):
            data = generate_smart_meter_data(
                self.duration, self.sampling_rate, APPLIANCES_UKDALE, APPLIANCE_PROFILES_UKDALE
            )
            self.household_data[h] = data
            self.grid_data += data['aggregate']

    def analyze(self):
        self.results = {
            'per_household': {},
            'grid_level': {},
            'demand_response': {},
        }

        for h, data in self.household_data.items():
            signal = data['aggregate']
            n = min(len(signal), 10000)
            preprocessed = preprocess_signal(signal[:n], {'outlierRemoval': {'threshold': 3}, 'smoothing': {'frameLength': 11, 'polyOrder': 3}})
            self.results['per_household'][h] = {
                'mean_power': float(np.mean(signal)),
                'max_power': float(np.max(signal)),
                'std_power': float(np.std(signal)),
                'n_outliers': preprocessed['nOutliers'],
            }

        self.results['grid_level'] = {
            'total_mean': float(np.mean(self.grid_data)),
            'total_max': float(np.max(self.grid_data)),
            'total_std': float(np.std(self.grid_data)),
            'num_households': self.num_households,
        }

        hourly_load = np.zeros(24)
        samples_per_hour = int(3600 / self.sampling_rate)
        for h in range(24):
            start = h * samples_per_hour
            end = min(start + samples_per_hour, len(self.grid_data))
            hourly_load[h] = np.mean(self.grid_data[start:end])

        peak_hour = int(np.argmax(hourly_load))
        off_peak_hour = int(np.argmin(hourly_load))
        peak_load = float(hourly_load[peak_hour])
        off_peak_load = float(hourly_load[off_peak_hour])

        self.results['demand_response'] = {
            'hourly_load': hourly_load.tolist(),
            'peak_hour': peak_hour,
            'off_peak_hour': off_peak_hour,
            'peak_load': peak_load,
            'off_peak_load': off_peak_load,
            'peak_off_peak_ratio': peak_load / (off_peak_load + 1e-12),
        }

        return self.results

    def visualize(self):
        return self.results

    def export_to_grid(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Results', 'grid_export.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        export_data = {
            'num_households': self.num_households,
            'grid_level': self.results.get('grid_level', {}),
            'demand_response': self.results.get('demand_response', {}),
        }
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        return path
