''' Simple Kalman filter based target tracker for a single passive radar
    target. Mostly intended as a simplified demonstration script. For better
    performance, use multitarget_kalman_tracker.py'''

import numpy as np
import matplotlib.pyplot as plt
import zarr
from tqdm import tqdm

from passiveRadar.config import getConfiguration
from passiveRadar.target_detection import simple_target_tracker_with_position
from passiveRadar.target_detection import simple_target_tracker
from passiveRadar.target_detection import CFAR_2D
from passiveRadar.target_detection import kalman_update_with_position
from passiveRadar.target_detection import kalman_update_with_doa_and_elevation
from passiveRadar.target_detection import plot_target_trajectory

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


def simple_dkr_tracker(config, xambg):
    print("Loaded range-doppler maps.")
    Nframes = xambg.shape[2]
    print("Applying CFAR filter...")
    
    CF = np.zeros(xambg.shape)
    for i in tqdm(range(Nframes)):
        CF[:, :, i] = CFAR_2D(xambg[:, :, i], 18, 4)

    print("Applying Kalman Filter...")
    # history = simple_target_tracker(CF, config['max_range_actual'], config['max_doppler_actual'])
    history = simple_target_tracker_with_position(CF, config['max_range_actual'], config['max_doppler_actual'])

    estimate = history['estimate']
    lockMode = history['lock_mode']
    
    trajectory = []  # List to store the target's trajectory (x, y)

    # 获取初始的发射机和接收机坐标
    prev_x_tx, prev_y_tx = 0, 0  # 发射机初始位置
    prev_x_rx, prev_y_rx = 100, 0  # 接收机初始位置
    
    for i in range(Nframes):
        # Process each frame to update the target state
        updated_position, new_state = kalman_update_with_position(estimate[i], history[i],time_step=i)  # Example coordinates
        trajectory.append(updated_position)

        # 更新发射机和接收机位置（这里假设它们的位置会随着时间变化）
        # 这里只是示例，实际应用中需要根据实际情况动态更新位置
        prev_x_tx, prev_y_tx = new_state[4], new_state[5]
        prev_x_rx, prev_y_rx = new_state[6], new_state[7]
    
    plot_target_trajectory(trajectory)

def simple_tracker_with_doa(config, xambg):
    print("Loaded range-doppler maps.")
    Nframes = xambg.shape[2]
    print("Applying CFAR filter...")
    
    CF = np.zeros(xambg.shape)
    for i in tqdm(range(Nframes)):
        CF[:, :, i] = CFAR_2D(xambg[:, :, i], 18, 4)

    print("Applying Kalman Filter with DOA and Elevation...")
    # 使用带有 DOA 和仰角的目标跟踪
    history = simple_target_tracker_with_position(CF, config['max_range_actual'], config['max_doppler_actual'])

    estimate = history['estimate']
    lockMode = history['lock_mode']
    
    trajectory = []  # List to store the target's trajectory (x, y)
    
    # 更新每一帧的目标状态和位置
    for i in range(Nframes):
        azimuth, elevation = estimate[i][0], estimate[i][1]  # 假设估计值包含 DOA 和仰角
        updated_position, new_state = kalman_update_with_doa_and_elevation([azimuth, elevation], history[i], time_step=i)
        trajectory.append(updated_position)
    
    plot_target_trajectory(trajectory)


if __name__ == "__main__":
    config = getConfiguration()

    xambgfile = config['range_doppler_map_fname']
    xambg = np.abs(zarr.load(xambgfile))

    simple_tracker(config, xambg)
