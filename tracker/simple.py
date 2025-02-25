''' Simple Kalman filter based target tracker for a single passive radar
    target. Mostly intended as a simplified demonstration script. For better
    performance, use multitarget_kalman_tracker.py'''

import numpy as np
import matplotlib.pyplot as plt
import zarr
from tqdm import tqdm

from passiveRadar.config import getConfiguration
from passiveRadar.target_detection import simple_target_tracker
from passiveRadar.target_detection import CFAR_2D


def simple_tracker(config, xambg):
    print("Loaded range-doppler maps.")
    Nframes = xambg.shape[2]
    print("Applying CFAR filter...")
    # CFAR filter each frame using a 2D kernel
    CF = np.zeros(xambg.shape)
    for i in tqdm(range(Nframes)):
        CF[:, :, i] = CFAR_2D(xambg[:, :, i], 18, 4)

    print("Applying Kalman Filter...")
    history = simple_target_tracker(
        CF, config['max_range_actual'], config['max_doppler_actual'])

    estimate = history['estimate']
    measurement = history['measurement']
    lockMode = history['lock_mode']

    unlocked = lockMode[:, 0].astype(bool)
    estimate_locked = estimate.copy()
    estimate_locked[unlocked, 0] = np.nan
    estimate_locked[unlocked, 1] = np.nan
    estimate_unlocked = estimate.copy()
    estimate_unlocked[~unlocked, 0] = np.nan
    estimate_unlocked[~unlocked, 1] = np.nan

    plt.figure(figsize=(12, 8))
    plt.plot(estimate_locked[:, 1],
             estimate_locked[:, 0], 'b', linewidth=3)
    plt.plot(
        estimate_unlocked[:, 1], estimate_unlocked[:, 0], c='r', linewidth=1, alpha=0.3)
    plt.xlabel('Doppler Shift (Hz)')
    plt.ylabel('Bistatic Range (km)')
    plt.show()


if __name__ == "__main__":
    config = getConfiguration()

    xambgfile = config['range_doppler_map_fname']
    xambg = np.abs(zarr.load(xambgfile))

    simple_tracker(config, xambg)
