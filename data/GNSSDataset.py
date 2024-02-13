import pandas as pd
import numpy as np
import torch
import glob
from tqdm import tqdm

class GNSSDataset(torch.utils.data.Dataset):
    def __init__(self, INPUT_PATH, max_seq_len=630):
        self.x_list = []
        self.y_list = []
        # max_satellites = 0
        filter_list = ["ArrivalTimeNanosSinceGpsEpoch", "RawPseudorangeMeters", "RawPseudorangeUncertaintyMeters", "IsrbMeters", "IonosphericDelayMeters", "TroposphericDelayMeters", "WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]

        for dirname in tqdm(sorted(glob.glob(f'{INPUT_PATH}/train/*/*'))[:2]):
            drive, phone = dirname.split('/')[-2:]
            tripID  = f'{drive}/{phone}'
            print(tripID)
            gnss_df = pd.read_csv(f'{dirname}/device_gnss.csv')            
            
            filtered = gnss_df[filter_list]
            sequence = self.process_gnss_sequence(filtered, max_seq_len, gnss_df)
            self.x_list.append(sequence)

            # get the label
            gt_df = pd.read_csv(f'{dirname}/ground_truth.csv')

            label_sequence = self.process_gt_sequence(gt_df)

            self.y_list.append(label_sequence)
            

    def __len__(self):
        return len(self.x_list)
    
    def __getitem__(self, idx):
        return self.x_list[idx], self.y_list[idx]
    
    def process_gnss_sequence(self, filtered, max_seq_len, gnss_df):
        sequence = []

        for time in gnss_df["utcTimeMillis"].unique():
            flattened = filtered[gnss_df["utcTimeMillis"] == time].to_numpy(dtype=np.float32).flatten()
            
            # fill zero 
            np.nan_to_num(flattened, copy=False, nan=0.0,)
            
            pad_to_len = np.pad(flattened, pad_width = (0, (max_seq_len-flattened.shape[0])), mode='constant', constant_values = (0,))
            sequence.append(pad_to_len)

        return np.array(sequence)
    
    def process_gt_sequence(self, gt_df):
        ## get the LatitudeDegrees and LongitudeDegrees
        times = gt_df["UnixTimeMillis"].unique()
        sequence = []
        for time in times:
            lat = gt_df[gt_df["UnixTimeMillis"] == time]["LatitudeDegrees"].to_numpy(dtype=np.float32)
            lon = gt_df[gt_df["UnixTimeMillis"] == time]["LongitudeDegrees"].to_numpy(dtype=np.float32)
            sequence.append(np.array([lat, lon]))
        return np.array(sequence)