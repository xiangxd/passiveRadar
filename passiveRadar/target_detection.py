''' target_detection.py: target detection tools for passive radar '''

import numpy as np
from passiveRadar.signal_utils import normalize
import scipy.signal as signal
import matplotlib.pyplot as plt


# create a data type to represent the Kalman filter's internals
# kalman_filter_dtype = np.dtype([
#     ('x' , np.float, (4, )),   # state estimate vector
#     ('P' , np.float, (4,4)),   # state estimate covariance matrix
#     ('F1', np.float, (4,4)),   # state transition model #1
#     ('F2', np.float, (4,4)),   # state transition model #2
#     ('Q' , np.float, (4,4)),   # process noise covariance matrix
#     ('H' , np.float, (2,4)),   # measurement matrix
#     ('R' , np.float, (2,2)),   # measurement noise covariance matrix
#     ('S' , np.float, (2,2))])  # innovation covariance matrix

# Kalman filter state (包含目标的 x, y 位置，目标的速度，以及发射机和接收机的坐标)
kalman_filter_dtype = np.dtype([
    ('x', np.float, (6,)),  # 扩展状态向量 [目标位置x, 目标位置y, 目标速度x, 目标速度y, 发射机位置x, 发射机位置y, 接收机位置x, 接收机位置y]
    ('P', np.float, (6, 6)),  # 状态协方差矩阵
    ('F1', np.float, (6, 6)),  # 状态转移矩阵 #1
    ('F2', np.float, (6, 6)),  # 状态转移矩阵 #2
    ('Q', np.float, (6, 6)),  # 过程噪声协方差矩阵
    ('H', np.float, (2, 6)),  # 测量矩阵 (只测量目标的位置)
    ('R', np.float, (2, 2)),  # 测量噪声协方差矩阵
    ('S', np.float, (2, 2))])  # 创新协方差矩阵

def kalman_update(measurement, currentState):
    ''' The standard Kalman filter update algorithm
    
    Parameters: 
        measurement:     measurement vector for current time step
        currentState:    kalman_filter_dtype containing the current filter state
    Returns:
        estimate:        the new estimate of the system state
        newState:        the new filter state 
    '''

    x  = currentState['x']  # state estimate vector
    P  = currentState['P']  # state estimate covariance matrix
    F1 = currentState['F1'] # state transition model #1
    F2 = currentState['F2'] # state transition model #2
    Q  = currentState['Q']  # process noise covariance matrix
    H  = currentState['H']  # measurement matrix
    R  = currentState['R']  # measurement noise covariance matrix
    S  = currentState['S']  # innovation covariance matrix

    # update state according to state transition model #1
    x = F1 @ x
    # update state covariance according to state transition model #2
    P = F2 @ P @ F2.T + Q
    # compute the innovation covariance
    S = H @ P @ H.T +  R
    # compute the optimal Kalman gain
    K = P @ H.T @ np.linalg.inv(S)
    # get the measurement
    Z = measurement
    # compute difference between prediction and measurement
    y = Z - H @ x
    # update the filter state
    x = x + K @ y
    # update the state covariance
    P = (np.eye(4) - K @ H) @ P

    # save the a posteriori state estimate
    estimate  = H @ x
    # construct the new filter state
    newState = (x, P, F1, F2, Q, H, R, S)

    return estimate, newState

def adaptive_kalman_update(measurement, lastMeasurement, currentState):
    ''' The standard Kalman filter update algorithm with adaptive estimation
    of the measurement covariance matrix.
    
    Parameters: 
        measurement:     measurement vector for current time step
        lastMeasurement: measurement vector for previous time step
        currentState:    kalman_filter_dtype containing the current filter state
    Returns:
        estimate:        the new estimate of the system state
        newState:        the new filter state 
    '''

    x  = currentState['x']  # state estimate vector
    P  = currentState['P']  # state estimate covariance matrix
    F1 = currentState['F1'] # state transition model #1
    F2 = currentState['F2'] # state transition model #1
    Q  = currentState['Q']  # process noise covariance matrix
    H  = currentState['H']  # measurement matrix
    R  = currentState['R']  # measurement noise covariance matrix
    S  = currentState['S']  # innovation covariance matrix

    # Adaptive estimation of  the measurement covariance matrix. Here I am 
    # using the squared distance between the current measurement and the 
    # previous measurement. This is very ad hoc but,,,, it seems to work well
    delta_meas = np.squeeze(measurement - lastMeasurement)
    R_scaling_factor = (delta_meas[0]**2 + delta_meas[1]**2)
    # R_estimate = delta_meas.T @ np.linalg.inv(S) @ delta_meas

    # update state according to state transition model #1
    x = F1 @ x
    # update state covariance according to state transition model #2
    P = F2 @ P @ F2.T + Q
    # compute the innovation covariance
    S = H @ P @ H.T +  R*R_scaling_factor
    # compute the optimal Kalman gain
    K = P @ H.T @ np.linalg.inv(S)
    # get the measurement
    Z = measurement
    # compute difference between prediction and measurement
    y = Z - H @ x
    # update the filter state
    x = x + K @ y
    # update the state covariance
    P = (np.eye(4) - K @ H) @ P

    # save the a posteriori state estimate
    estimate  = H @ x
    # construct the new filter state
    newState = (x, P, F1, F2, Q, H, R, S)

    return estimate, newState

def kalman_extrapolate(currentState):
    ''' Update Kalman filter state according to internal model (use when no 
    measurements are available)
    
    Parameters: 
        currentState:  kalman_filter_dtype containing the filter state at time t
    Returns:
        estimate:      the estimate of the system state for time t+1
        newState:      the filter state for time t+1 
    '''

    x  = currentState['x']  # state estimate vector
    P  = currentState['P']  # state estimate covariance matrix
    F1 = currentState['F1'] # state transition model #1
    F2 = currentState['F2'] # state transition model #1
    Q  = currentState['Q']  # process noise covariance matrix
    H  = currentState['H']  # measurement matrix
    R  = currentState['R']  # measurement noise covariance matrix
    S  = currentState['S']  # innovation covariance matrix

    # update state according to state transition model #1
    x = F1 @ x
    # update state covariance according to state transition model #2
    P = F2 @ P @ F2.T + Q
    # compute the innovation covariance
    S = H @ P @ H.T +  R

    # save the a posteriori state estimate
    estimate  = H @ x
    # construct the new filter state
    newState = (x, P, F1, F2, Q, H, R, S)

    return estimate, newState

# A datatype that represents a single target track 
target_track_dtype = np.dtype([
    ('status', np.int),     # the status of the track is 0, 1 or 2.
                            # 0: free (no target assigned to this track)
                            # 1: tracking preliminary target 
                            # 2: tracking confirmed target
    ('lifetime', np.int),   # keeps track of how long the track has been alive
    ('measurement', np.float, (2,)), # measurement for the current timestep
    ('estimate', np.float, (2,)),    # state estimate for the current timestep
    ('measurement_history', np.float, (20,)), # over the past 20 timesteps, keep
    # track of where there were confirmed measurements assigned to this track.
    # This is used to determine when a track has lost its target
    ('kalman_state', kalman_filter_dtype)]) # the state of the Kalman filter

def get_measurements(dataFrame, p, frame_extent):
    ''' extract a list of candidate measurements from a range-doppler map frame

    Parameters:

        dataFrame:      2D array containting the range-doppler map to acquire
                        measurements from.
        p:              Detection threshold. Pixels whose amplitudes are in the 
                        upper pth percentile of the values in dataFrame are 
                        added to the list of candidate measurements.
        frame_extent:   Defines the edges lengths for  the input frame allowing
                        pixel indices to be converted to measurement values.
    Returns:

        candidateRange:     Vector containing all the range values for the M 
                            candidate measurements
        candidateDoppler:   Vector containing all the Doppler values for the M 
                            candidate measurements
        candidateStrength:  Vector containing all the pixel amplitudes values 
                            for the M candidate measurements
    '''

    # define the extent of the measurement region
    rangeExtent   = frame_extent[1]   # km
    dopplerExtent = frame_extent[0]   # Hz
    rpts = np.linspace(rangeExtent, 0, dataFrame.shape[1])
    dpts = np.linspace(-1*dopplerExtent, dopplerExtent, dataFrame.shape[0])
    rangeBinCenters   = np.expand_dims(rpts, 1)
    dopplerBinCenters = np.expand_dims(dpts, 0)

    rangeBinCenters   = np.tile(rangeBinCenters,    (1, dataFrame.shape[0]))
    dopplerBinCenters = np.tile(dopplerBinCenters,  (dataFrame.shape[1], 1))

    # normalize input frame
    dataFrame = dataFrame/np.mean(np.abs(dataFrame).flatten())
    # get the orientation right :))
    dataFrame = np.fliplr(dataFrame.T)
    # zero out the problematic sections where there is persistent
    # clutter (small range / doppler)
    dataFrame[:8, :] = 0
    dataFrame[-8:, :] = 0
    dopplerCenter = dataFrame.shape[1]//2
    dataFrame[:, dopplerCenter-4:dopplerCenter+4] = 0

    # calculate the detection threshold. There are 300x512 = 153600 pixels per 
    # range doppler frame, so taking 99.8th percentile selects the strongest 300
    # or so to be used as potential measurements.
    threshold = np.percentile(dataFrame, 99.8)

    # find points on the input frame where amplitude exceeds threshold
    candidateMeasIdx    = np.nonzero(dataFrame >= threshold)

    # extract candidate measurement positions and measurement strength
    candidateRange      = rangeBinCenters[candidateMeasIdx]
    candidateDoppler    = dopplerBinCenters[candidateMeasIdx]
    candidateStrength   = dataFrame[candidateMeasIdx]

    # sort the candidate measurements in decreasing order of strength
    sort_idx = np.flip(np.argsort(candidateStrength))
    candidateRange      = candidateRange[sort_idx]
    candidateDoppler    = candidateDoppler[sort_idx]
    candidateStrength   = candidateStrength[sort_idx]

    candidateMeas = np.stack((candidateRange, candidateDoppler, candidateStrength))

    return candidateMeas

def associate_measurements(trackState, candidateMeas):
    '''Associate a set of candidate measurements with a target track
    
    Parameters:
        trackState:     target_track_dtype containing the current track state
        candidateMeas:  list of candidate measurements
    Returns:
        newMeasurement: the measurement selected to update this target track. 
                        Returns None if there are no suitable measurements. 
        candidateMeas:  The updated list of candidate measurements. If a 
                        measurement has been selected for this target then 
                        measurements close to the selected measurement are 
                        removed.
    '''
    
    track_status = trackState['status']

    # get the last measurement and state estimate for this track
    lastRangeMeas    = trackState['measurement'][0]
    lastDopplerMeas  = trackState['measurement'][1]
    lastDopplerEst  = trackState['estimate'][1]
    lastRangeEst    = trackState['estimate'][0]

    candidateRange      = candidateMeas[0, :]
    candidateDoppler    = candidateMeas[1, :]
    candidateStrength   = candidateMeas[2, :]

    ####################### FIRST VALIDATION STEP ##############################

    if track_status == 0:
        # if the track state is free we're not picky, we consider any measurement
        earlyGate = np.ones(candidateRange.shape).astype(bool)

    elif track_status == 1:
        # if the track state is preliminary we look within 5km and 12Hz of the 
        # last confirmed measurement
        rangeGate   = (np.abs(candidateRange   - lastRangeMeas)   < 5)
        dopplerGate = (np.abs(candidateDoppler - lastDopplerMeas) < 24)
        earlyGate =  rangeGate.astype(bool) & dopplerGate.astype(bool)

    else:
        # if the track state is confirmed we look within 4km and 10Hz of the 
        # last state estimate
        rangeGate = (np.abs(candidateRange - lastRangeEst) < 4)
        dopplerGate = (np.abs(candidateDoppler - lastDopplerEst) < 20)
        earlyGate =  rangeGate.astype(bool) & dopplerGate.astype(bool)
    
    rangeMeas       = candidateRange[earlyGate]
    dopplerMeas     = candidateDoppler[earlyGate]
    strengthMeas    = candidateStrength[earlyGate]

    ############ SECOND VALIDATION STEP (ONLY FOR CONFIRMED TRACKS) ############

    if track_status == 2: # confirmed tracks
        # apply a stricter validation gate based on the innovation
        # covariance matrix of the kalman filter for this track
        NCloseCandidates = np.sum(earlyGate)
        validationGate = np.zeros((NCloseCandidates,)).astype(bool)
        S = trackState['kalman_state']['S']
        for kk in range(NCloseCandidates):
            rangeDiff = lastRangeMeas - rangeMeas[kk]
            dopplerDiff = lastDopplerMeas - dopplerMeas[kk]
            zerr = np.array([rangeDiff, dopplerDiff])
            validationGate[kk] = zerr.T @ np.linalg.inv(S) @ zerr < 6
            
        # Get the measurements that fit the final validation criteria for 
        # this track
        rangeMeas       = rangeMeas[validationGate]
        dopplerMeas     = dopplerMeas[validationGate]
        strengthMeas    = strengthMeas[validationGate]

    ############################################################################
    
    # how many measurements did we get?
    measurementsFound = rangeMeas.size

    if measurementsFound == 0:
        # if there are no suitable measurements for this track we return None
        # and leave the list of candidate measurements unchanged
        return None, candidateMeas

    elif measurementsFound > 1:
        # if there are multiple measurements that match with this target we need
        # to pick which one to use
        if track_status == 0:
            # take the strongest measurement if the target is unconfirmed
            rangeMeas       = candidateRange[0]
            dopplerMeas     = candidateDoppler[0]
            strengthMeas    = candidateStrength[0] 
            newMeasurement = np.squeeze(np.array([rangeMeas, dopplerMeas]))   
        
            rangeGate   = (np.abs(candidateRange - rangeMeas) < 10).astype(bool)
            dopplerGate = (np.abs(candidateDoppler - dopplerMeas) < 12).astype(bool)
            earlyGate   =  rangeGate & dopplerGate
        elif track_status==1:
            # if there are multiple candidate measurements pick the nearest one
            # (Nearest Neighbour Standard Filter). I should definitely screw
            # around with various other methods here eg PDAF
            ixm = np.argmin(np.sqrt(rangeMeas**2 + dopplerMeas**2))
            rangeMeas       = rangeMeas[ixm]
            dopplerMeas     = dopplerMeas[ixm]
            strengthMeas    = strengthMeas[ixm]
        if track_status == 2:
            # # if there are multiple candidate measurements pick the strongest one
            # # (Strongest Neighbour Standard Filter). 
            rangeMeas       = rangeMeas[0]
            dopplerMeas     = dopplerMeas[0]
            strengthMeas    = strengthMeas[0]
            newMeasurement = np.squeeze(np.array([rangeMeas, dopplerMeas]))

    candidateRange      = candidateRange[~earlyGate]
    candidateDoppler    = candidateDoppler[~earlyGate]
    candidateStrength   = candidateStrength[~earlyGate]

    newMeasurement = np.squeeze(np.array([rangeMeas, dopplerMeas])) 
    candidateMeas = np.stack((candidateRange, candidateDoppler, candidateStrength))

    return newMeasurement, candidateMeas

def initialize_track(measurement):
    '''Create a new target track with default parameter values

    Parameters:
        measurement:            first measurement for this target track
            if measurement is None, the track is initialized in the 'free' state
            (status = 0) at an arbitrary set of coordinates
            if measurement is a set of coordinates, the track is initialized at 
            these coordinates in the 'preliminary' state (status = 1)
    Returns:    
        initialTrackerState:    target_track_dtype containing the initialized
                                target track
    '''
    if measurement is None:
        r = 0 # initialize position to 0 (this is arbitrary)
        f = 0
        status=0
    else:
        r = measurement[0]
        f = measurement[1]
        status = 1

    # create initial Kalman filter parameters
    # the algorithm is pretty robust to changes in these values, the numbers
    # given here seem to work well
    x = np.array([r, 0, f, -1])
    P = np.diag([5.0, 0.0225, 0.04, 0.1])
    F1 = np.array([[1,0,-0.003, 0], [0, 0,-0.003,-0.003], [0,0,1,1], [0,0,0,1]])
    F2 = np.array([[1,1,0,0], [0,1,0,0], [0,0,1,1], [0,0,0,1]])
    Q = np.diag([4.0, 0.03, 0.2, 0.08])
    H = np.array([[1,0,0,0], [0,0,1,0]])
    R = np.diag([5, 2])
    S = np.diag([1, 1])

    # initialize the target tracker state
    lifetime            = 1
    estimate            = H @ x
    measurement         = np.array([r, f])
    targetHistory       = np.zeros((20,))
    targetHistory[0]    = 1
    targetHistory[5:10] = 1
    kalmanState         = (x, P, F1, F2, Q, H, R, S)
    
    initialTrackState = np.array([(status, lifetime, estimate, measurement, 
        targetHistory, kalmanState)], dtype=target_track_dtype)

    return initialTrackState

def update_track(currentState, newMeasurement):
    ''' Update a single target track with a new measurement.
    
    Parameters:
        currentState:   target_track_dtype containting the current state of the 
                        target track
        newMeasurement: measurement vector for this timestep
    Returns:
        newState:       target_track_dtype containing the updated track state
    '''

    status = currentState['status']
    lifetime = np.squeeze(currentState['lifetime'])
    measurement = currentState['measurement']
    kalmanState = currentState['kalman_state']
    targetFoundHistory = currentState['measurement_history'].flatten()

    if newMeasurement is None:
        # no suitable measurements have been assigned to this track.
        # extrapolate the next state from the current state estimate.
        newEstimate, newKalmanState = kalman_extrapolate(kalmanState)
        newTargetFoundHistory = np.concatenate(([0], targetFoundHistory[:-1]))
        newMeasurement = measurement # just keep the last confirmed measurement

    else:

        # use the Kalman filter to update the state estimate
        newEstimate, newKalmanState = adaptive_kalman_update(newMeasurement, 
                                        measurement, kalmanState)
        newTargetFoundHistory = np.concatenate(([1], targetFoundHistory[:-1]))

    
    # if the track status is currently 1 (preliminary), we can decide to either
    # promote it to a confirmed target track or kill it depending on how many
    # measurements consistent with the estimated target state have been obtained 
    # over the past few timesteps
    if status == 1:
        # condition to kill track
        if (lifetime > 4 ) and (np.sum(targetFoundHistory[0:10]) < 6):
            status = 0
        # condition to promote to confirmed target track
        if (lifetime > 4 ) and (np.sum(targetFoundHistory[0:10]) > 8):
            status = 2

    # If the track status is confirmed we can kill it if not enough measurements
    # have been found.
    elif status == 2:
        # condition to kill track
        if (lifetime > 4) and (np.sum(targetFoundHistory) < 4):
            status = 0

    # construct the new track state
    newState = np.array([(status, lifetime+1, newMeasurement, newEstimate, 
        newTargetFoundHistory, newKalmanState)], dtype=target_track_dtype)

    return newState

def multitarget_tracker(data, frame_extent, N_TRACKS):
    ''' Radar target tracker for multiple targets. 
    
    Parameters:
        data:           3D array containting a stack of range-doppler map frames

        frame_extent:   Defines the edges lengths for  the input frame allowing
                        pixel indices to be converted to measurement values.

        N_TRACKS:       Number of tracks. Corresponds to the maximum number of
                        targets that can be tracked simultaneously
    Returns:
        history:        (Nframes, N_TRACKS) array of target_track_dtype.
                        Contains the complete state of each of the target tracks
                        at each time step
    '''

    # number of data frames
    Nframes = data.shape[2]

    # initialize a vector of tatget tracks
    trackStates  = np.empty((N_TRACKS,), dtype=target_track_dtype)
    for i in range(trackStates.shape[0]):
        # start each track in the unlocked state
        trackStates[i] =  initialize_track(None)

    # make a storage array for the results at each timestep
    tracker_history = np.empty((Nframes, N_TRACKS), dtype = target_track_dtype)

    # loop over input frames
    for i in range(Nframes):

        # get the range-doppler frame for this timestep
        dataFrame = data[:,:,i]

        # get the new list of candidate measurements for this frame
        candidateMeas = get_measurements(dataFrame, 99.8, frame_extent)

        # find out which tracks are in the confirmed, preliminary and free states
        confirmedTracks = np.argwhere(trackStates['status'] == 2).flatten()
        prelimTracks    = np.argwhere(trackStates['status'] == 1).flatten()
        freeTracks      = np.argwhere(trackStates['status'] == 0).flatten()

        for track_idx in confirmedTracks:
            # loop over the confirmed tracks first (confirmed tracks get first
            # access to the candidate measurements.)
            trackState = trackStates[track_idx]
            newMeasurement, candidateMeas = associate_measurements(trackState, candidateMeas)
            newTrackState = update_track(trackState, newMeasurement)
            trackStates[track_idx] = newTrackState

        for track_idx in prelimTracks:
            # Next update the preliminary tracks
            trackState = trackStates[track_idx]
            newMeasurement, candidateMeas = associate_measurements(trackState, candidateMeas)
            newTrackState = update_track(trackState, newMeasurement)
            trackStates[track_idx] = newTrackState

        for track_idx in freeTracks:
            # Finally, assign free tracks to remaining measurements
            trackState = trackStates[track_idx]
            if candidateMeas.size == 0:
                print("no more measurements available, track remaining free")
                break
            newMeasurement, candidateMeas = associate_measurements(trackState, candidateMeas)
            newTrackState = initialize_track(newMeasurement)
            trackStates[track_idx] = newTrackState

        # save all the track states for this timestep
        tracker_history[i,:] = trackStates

    return tracker_history

# create a data type that represents the internal state of the 
# target tracker
target_track_dtype_simple = np.dtype([
    ('lock_mode',       np.float, (4,)),
    ('measurement',     np.float, (2,)),
    ('measurement_idx', np.int,   (2,)),
    ('estimate',        np.float, (2,)),
    ('range_extent',    np.float),
    ('doppler_extent',  np.float),
    ('kalman_state',    kalman_filter_dtype)])

def simple_track_update(currentState, inputFrame):
    ''' Update step for passive radar target tracker.
    
    Parameters:
        currentState: target_track_dtype_simple_simple containting the current state
        of the target tracker
        inputFrame: range-doppler map from which to acquire a measurement
    Returns:
        newState: target_track_dtype_simple_simple containing the updated target tracker
        state'''

    # Get the current tracker state
    lockMode        = currentState['lock_mode'][0]
    measurement     = currentState['measurement'][0]
    measIdx         = currentState['measurement_idx'][0]
    estimate        = currentState['estimate'][0]
    rangeExtent     = currentState['range_extent'][0]
    dopplerExtent   = currentState['doppler_extent'][0]
    kalmanState     = currentState['kalman_state'][0]

    # Now based on the current tracker state we choose where on the
    # input frame to look for our new measurement. If the tracker is 
    # in an unlocked state we look for the target anywhere. If it is in 
    # one of the target-locked states we restrict attention to a 
    # rectangle around the previous measurement. The size of the
    # rectangle depends on which of the target-locked states we're in.

    lx = measIdx[1]
    ly = measIdx[0]

    # first lock state (initial lock on)
    if (lockMode[1] == 1):
        measurementGate = np.zeros(inputFrame.shape)
        measurementGate[ly-24:ly+24,lx-48:lx+48] = 1.0
        inputFrame = inputFrame * measurementGate

    # second lock state (fully locked)
    elif (lockMode[2] == 1):
        measurementGate = np.zeros(inputFrame.shape)
        measurementGate[ly-16:ly+16,lx-32:lx+32] = 1.0
        inputFrame = inputFrame * measurementGate
        
    # third lock state (losing lock)
    elif (lockMode[3] == 1):
        measurementGate = np.zeros(inputFrame.shape)
        measurementGate[ly-24:ly+24,lx-48:lx+48] = 1.0
        inputFrame = inputFrame * measurementGate

    # else unlocked (don't apply a measurement gate)

    # obtain the new measurement by finding the indices of the 
    # maximum amplitude point of the range-doppler map
    newMeasIdx = np.unravel_index(np.argmax(inputFrame), inputFrame.shape)

    # convert the indices to range-doppler values
    range_meas = rangeExtent*(1 - newMeasIdx[0]/inputFrame.shape[0])
    doppler_meas = dopplerExtent*(2*newMeasIdx[1]/inputFrame.shape[1] - 1)

    # construct the new measurement vector
    newMeasurement = np.array([range_meas, doppler_meas])

    # check if we seem to be locked on to a target. If the new measurement
    # is close to the estimate from the last time step then we assume that
    # a target has been found and we are pleased about it
    surprise_level = newMeasurement - estimate
    badnessMetric = (surprise_level[0]**2 + (0.5*surprise_level[1])**2)**0.5
    targetFound    = badnessMetric < 12

    if targetFound:
        # matrix encodes state update rules for target found
        track_update_matrix = np.array([[0,1,0,0], [0,0,1,0], [0,0,1,0],[0,0,1,0]]).T
    else:
        # matrix encodes state update rules for target not found
        track_update_matrix = np.array([[1,0,0,0], [1,0,0,0], [0,0,0,1],[1,0,0,0]]).T
    
    # update the tracker state
    newLockMode = track_update_matrix @ lockMode

    # use the Kalman filter to update the state estimate
    newEstimate, newKalmanState = adaptive_kalman_update(newMeasurement, measurement, kalmanState)

    newState = np.array([(newLockMode, newMeasurement, newMeasIdx, 
        newEstimate, rangeExtent, dopplerExtent, newKalmanState)], dtype = target_track_dtype_simple)

    return newState


def simple_target_tracker(data, rangeExtent, dopplerExtent):

    N_measurements = data.shape[2]

    # create initial values for the Kalman Filter
    # initial values for P and Q are taken from P.E. Howland et al, "FM radio 
    # based bistatic radar"

    x = np.array([30, 2, -20, -1])
    P = np.diag([5.0, 0.0225, 0.04, 0.1])
    F1 = np.array([[1,0,-0.003, 0], [0, 0,-0.003,-0.03], [0,0,1,1], [0,0,0,1]])
    F2 = np.array([[1,1,0,0], [0,1,0,0], [0,0,1,1], [0,0,0,1]])
    Q = np.diag([2.0, 0.02, 0.2, 0.05])
    H = np.array([[1,0,0,0], [0,0,1,0]])
    R = np.diag([5, 5])
    S = np.diag([1, 1])

    # initialize the target tracker state
    lockMode      = np.array([1,0,0,0])    # start in unlocked state
    estimate      = H @ x
    measurement   = np.array([35.0, -30.0]) # random IC
    measIdx       = np.array([50,50], dtype = np.int) # random IC
    # rangeExtent   = 375 # km
    # dopplerExtent = 256/1.092 # Hz
    kalmanState   = np.array([(x, P, F1, F2, Q, H, R, S)], dtype=kalman_filter_dtype)
    
    trackerState = np.array([(lockMode, estimate, measurement,
        measIdx, rangeExtent, dopplerExtent, kalmanState)], dtype = target_track_dtype_simple)

    # preallocate an array to store the tracking results
    history = np.empty((N_measurements,), dtype = target_track_dtype_simple)

    # do the tracking!!111
    for i in range(N_measurements):

        # get a frame of the range-doppler map
        dataFrame = data[:,:,i]

        # normalize it
        dataFrame = dataFrame/np.mean(np.abs(dataFrame).flatten())

        # get the orientation right :))
        dataFrame = np.fliplr(dataFrame.T)

        # zero out the problematic sections (small range / doppler)
        dataFrame[:8, :] = 0
        dataFrame[-8:, :] = 0
        dataFrame[:,250:260] = 0

        # do the update step
        trackerState = simple_track_update(trackerState, dataFrame)
        
        # save the results
        history[i] = trackerState
    
    return history

def CFAR_2D(X, fw, gw, thresh = None):
    '''constant false alarm rate target detection
    
    Parameters:
        fw: CFAR kernel width 
        gw: number of guard cells
        thresh: detection threshold
    
    Returns:
        X with CFAR filter applied'''

    Tfilt = np.ones((fw,fw))/(fw**2 - gw**2)
    e1 = (fw - gw)//2
    e2 = fw - e1 + 1
    Tfilt[e1:e2, e1:e2] = 0

    CR = normalize(X) / (signal.convolve2d(X, Tfilt, mode='same', boundary='wrap') + 1e-10)
    if thresh is None:
        return CR
    else:
        return CR > thresh


def simple_target_tracker_with_position(data, x_tx, y_tx, x_rx, y_rx):
    N_measurements = data.shape[2]
    trackStates = np.empty((N_measurements,), dtype=kalman_filter_dtype)

    # Initialize target tracks
    for i in range(trackStates.shape[0]):
        trackStates[i] = initialize_track(None)

    # Store tracking history (including position)
    tracker_history = np.empty((N_measurements,), dtype=kalman_filter_dtype)

    for i in range(N_measurements):
        dataFrame = data[:, :, i]

        # Normalize the data frame and get new measurements
        dataFrame = dataFrame / np.mean(np.abs(dataFrame).flatten())
        dataFrame = np.fliplr(dataFrame.T)

        # Extract Bistatic Range from measurements (for simplicity, use the first frame)
        measurement = np.array([50, 60])  # Example measurement for testing

        # Update the track with the new position estimate
        updatedEstimate, newState = kalman_update_with_position(measurement, trackStates[i], x_tx, y_tx, x_rx, y_rx)

        # Save the updated state
        trackStates[i] = newState
        tracker_history[i] = updatedEstimate

    return tracker_history


# 计算 Bistatic 交点 (笛卡尔坐标)
def bistatic_to_cartesian(R_tx, R_rx, x_tx, y_tx, x_rx, y_rx):
    """
    计算目标的笛卡尔坐标，基于 Bistatic Range 和目标到接收机/发射机的距离
    :param R_tx: 目标到发射机的距离
    :param R_rx: 目标到接收机的距离
    :param x_tx: 发射机的 x 坐标
    :param y_tx: 发射机的 y 坐标
    :param x_rx: 接收机的 x 坐标
    :param y_rx: 接收机的 y 坐标
    :return: 目标的笛卡尔坐标 (x, y)
    """
    # 假设目标位于平面上，通过解方程求目标位置
    A = 2 * (x_tx - x_rx)
    B = 2 * (y_tx - y_rx)
    C = R_rx**2 - R_tx**2 - x_rx**2 + x_tx**2 - y_rx**2 + y_tx**2

    # 通过线性代数计算 x 和 y 的位置
    x = C / A
    y = (R_tx**2 - (x - x_tx)**2)**0.5

    return x, y

# # 示例调用
# x_tx, y_tx = 0, 0  # 发射机位置
# x_rx, y_rx = 100, 0  # 接收机位置
# R_tx = 50  # 目标到发射机的距离
# R_rx = 60  # 目标到接收机的距离

# target_x, target_y = bistatic_to_cartesian(R_tx, R_rx, x_tx, y_tx, x_rx, y_rx)
# print(f"Target Position (Cartesian): x = {target_x}, y = {target_y}")

def bistatic_to_cartesian_3d(R_tx, R_rx, x_tx, y_tx, z_tx, x_rx, y_rx, z_rx):
    """
    计算目标的三维笛卡尔坐标，基于 Bistatic Range 和目标到接收机/发射机的距离。
    
    :param R_tx: 目标到发射机的距离
    :param R_rx: 目标到接收机的距离
    :param x_tx: 发射机的 x 坐标
    :param y_tx: 发射机的 y 坐标
    :param z_tx: 发射机的 z 坐标
    :param x_rx: 接收机的 x 坐标
    :param y_rx: 接收机的 y 坐标
    :param z_rx: 接收机的 z 坐标
    :return: 目标的三维笛卡尔坐标 (x, y, z)
    """
    # 计算目标位置的 x 和 y 坐标
    A = 2 * (x_tx - x_rx)
    B = 2 * (y_tx - y_rx)
    C = R_rx**2 - R_tx**2 - x_rx**2 + x_tx**2 - y_rx**2 + y_tx**2

    x = C / A
    y = (R_tx**2 - (x - x_tx)**2)**0.5

    # 计算目标位置的 z 坐标
    # 解方程来计算 z 坐标
    D = 2 * (z_tx - z_rx)
    E = R_rx**2 - R_tx**2 - z_rx**2 + z_tx**2

    z = E / D

    return x, y, z

# # 示例调用
# x_tx, y_tx, z_tx = 0, 0, 0  # 发射机位置
# x_rx, y_rx, z_rx = 100, 0, 0  # 接收机位置
# R_tx = 50  # 目标到发射机的距离
# R_rx = 60  # 目标到接收机的距离

# target_x, target_y, target_z = bistatic_to_cartesian_3d(R_tx, R_rx, x_tx, y_tx, z_tx, x_rx, y_rx, z_rx)
# print(f"Target Position (Cartesian): x = {target_x}, y = {target_y}, z = {target_z}")


# Example target tracker update function
def update_track_with_position(currentState, newMeasurement, x_tx, y_tx, x_rx, y_rx):
    """
    Update target tracker state with position calculation in Cartesian coordinates.
    
    :param currentState: Current state of the target track (including Kalman state)
    :param newMeasurement: New measurement for the current timestep
    :param x_tx: Transmitter x-coordinate
    :param y_tx: Transmitter y-coordinate
    :param x_rx: Receiver x-coordinate
    :param y_rx: Receiver y-coordinate
    :return: Updated target state with position in Cartesian coordinates
    """
    # Extract the current Kalman filter state and other information
    kalmanState = currentState['kalman_state']
    range_tx = newMeasurement[0]  # Bistatic Range to transmitter
    range_rx = newMeasurement[1]  # Bistatic Range to receiver

    # Convert from Bistatic Range to Cartesian coordinates
    target_x, target_y = bistatic_to_cartesian(range_tx, range_rx, x_tx, y_tx, x_rx, y_rx)
    
    # Here you can integrate the position with the Kalman filter update (if needed)
    # For example, update the Kalman filter state with the new position information

    # Simulating a simple Kalman filter update for position (just for illustration)
    # Update the Kalman filter state (this can be more complex based on your Kalman update logic)
    updatedEstimate = np.array([target_x, target_y])  # Updated estimate with Cartesian position
    
    # Example: Constructing the updated state
    updatedState = currentState.copy()
    updatedState['estimate'] = updatedEstimate  # Update the estimate with new position

    # Return the updated state
    return updatedState

# # Example usage
# currentState = {
#     'kalman_state': np.zeros(10),  # Kalman filter state
#     'estimate': np.array([0, 0]),  # Initial estimate of position
#     'measurement': np.array([50, 60]),  # Initial measurement (Bistatic Range)
# }

# # Assume these are the coordinates of the transmitter and receiver
# x_tx, y_tx = 0, 0  # Transmitter at origin
# x_rx, y_rx = 100, 0  # Receiver 100 km away on the x-axis

# newMeasurement = np.array([50, 60])  # New Bistatic Range measurement

# # Update the target state with new position
# updatedState = update_track_with_position(currentState, newMeasurement, x_tx, y_tx, x_rx, y_rx)

# print("Updated target position:", updatedState['estimate'])

def multitarget_tracker_with_position(data, frame_extent, N_TRACKS):
    Nframes = data.shape[2]
    trackStates = np.empty((N_TRACKS,), dtype=target_track_dtype)

    # Initialize target tracks
    for i in range(trackStates.shape[0]):
        trackStates[i] = initialize_track(None)

    # Store tracking history (now including position)
    tracker_history = np.empty((Nframes, N_TRACKS), dtype=target_track_dtype)

    # Loop through frames
    for i in range(Nframes):
        dataFrame = data[:, :, i]

        # Get candidate measurements for this frame
        candidateMeas = get_measurements(dataFrame, 99.8, frame_extent)

        for track_idx in range(N_TRACKS):
            trackState = trackStates[track_idx]
            newMeasurement, candidateMeas = associate_measurements(trackState, candidateMeas)
            
            # Update with new position using bistatic_to_cartesian
            updatedState = update_track_with_position(trackState, newMeasurement, x_tx, y_tx, x_rx, y_rx)
            trackStates[track_idx] = updatedState

        tracker_history[i, :] = trackStates

    return tracker_history

def kalman_update_with_position(measurement, currentState, prev_x_tx, prev_y_tx, prev_x_rx, prev_y_rx):
    """
    计算目标的笛卡尔坐标，并动态获取发射机和接收机的位置
    
    :param measurement: 当前测量值（Bistatic Range）
    :param currentState: 当前 Kalman 滤波器状态
    :param prev_x_tx: 上次发射机的 x 坐标
    :param prev_y_tx: 上次发射机的 y 坐标
    :param prev_x_rx: 上次接收机的 x 坐标
    :param prev_y_rx: 上次接收机的 y 坐标
    :return: 更新后的目标位置和状态
    """
    
    # 从当前状态中提取目标位置、速度、发射机和接收机位置
    x = currentState['x']
    P = currentState['P']
    F1 = currentState['F1']
    F2 = currentState['F2']
    Q = currentState['Q']
    H = currentState['H']
    R = currentState['R']
    S = currentState['S']

    # 提取发射机和接收机的坐标（从 Kalman 状态中提取）
    x_tx, y_tx, x_rx, y_rx = x[4], x[5], x[6], x[7]

    # 更新目标位置、速度（标准的 Kalman 滤波器步骤）
    x = F1 @ x
    P = F2 @ P @ F2.T + Q
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    
    Z = measurement
    y = Z - H @ x
    x = x + K @ y
    P = (np.eye(6) - K @ H) @ P

    # 动态更新发射机和接收机的位置
    # 假设接收机和发射机根据某些测量或运动模型进行动态更新
    # 这里使用示例值，你可以根据实际应用情况来调整
    updated_x_tx = prev_x_tx + 1  # 假设发射机位置变化（例如，沿 x 轴移动）
    updated_y_tx = prev_y_tx
    updated_x_rx = prev_x_rx + 1  # 假设接收机位置变化（例如，沿 x 轴移动）
    updated_y_rx = prev_y_rx

    # 使用 Bistatic Range 更新目标位置
    R_tx, R_rx = measurement
    target_x, target_y = bistatic_to_cartesian(R_tx, R_rx, updated_x_tx, updated_y_tx, updated_x_rx, updated_y_rx)
    
    updated_position = np.array([target_x, target_y])
    
    # 返回更新后的目标位置和状态
    newState = (x, P, F1, F2, Q, H, R, S)
    
    return updated_position, newState

def kalman_update_with_position(measurement, currentState, time_step):
    """
    Kalman filter update with position estimation.
    Kalman 滤波器更新，计算目标的笛卡尔坐标，并动态获取发射机和接收机的位置。
    
    :param measurement: Current measurement (Bistatic Range)
    :param currentState: Current Kalman filter state
    :param time_step: 当前时间步长，用于更新发射机和接收机位置
    :return: Updated Kalman filter state
    """
    # 从当前状态中提取目标位置、速度、发射机和接收机位置
    x = currentState['x']
    P = currentState['P']
    F1 = currentState['F1']
    F2 = currentState['F2']
    Q = currentState['Q']
    H = currentState['H']
    R = currentState['R']
    S = currentState['S']

    # 获取发射机和接收机的坐标（从 Kalman 状态中提取）
    x_tx, y_tx, x_rx, y_rx = x[4], x[5], x[6], x[7]
    
    # Perform Kalman filter prediction and update (standard steps)
    x = F1 @ x
    P = F2 @ P @ F2.T + Q
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    
    Z = measurement
    y = Z - H @ x
    x = x + K @ y
    P = (np.eye(6) - K @ H) @ P

    # Use Bistatic Range to estimate the target's Cartesian position
    R_tx, R_rx = measurement
    target_x, target_y = bistatic_to_cartesian(R_tx, R_rx, x_tx, y_tx, x_rx, y_rx)
    
    # Update the state estimate with the new position
    updatedEstimate = np.array([target_x, target_y])

    # 动态更新发射机和接收机的位置
    updated_x_tx, updated_y_tx, updated_x_rx, updated_y_rx = dynamic_position_update(x_tx, y_tx, x_rx, y_rx, time_step)
    
    # 更新 Kalman 状态中的发射机和接收机位置
    x[4], x[5], x[6], x[7] = updated_x_tx, updated_y_tx, updated_x_rx, updated_y_rx
  
    # Construct new state
    newState = (x, P, F1, F2, Q, H, R, S)
    
    return updatedEstimate, newState

def kalman_update_with_doa_and_elevation(measurement, currentState, time_step):
    """
    使用 DOA 和仰角来更新目标的位置和状态
    :param measurement: 目标的 DOA 和仰角
    :param currentState: 当前 Kalman 滤波器状态
    :param time_step: 当前时间步长
    :return: 更新后的目标位置和状态
    """
    
    # 从当前状态中提取目标位置、速度
    x = currentState['x']
    P = currentState['P']
    F1 = currentState['F1']
    F2 = currentState['F2']
    Q = currentState['Q']
    H = currentState['H']
    R = currentState['R']
    S = currentState['S']
    
    # 提取 DOA 和仰角信息
    azimuth, elevation = measurement[0], measurement[1]
    
    # 更新目标位置、速度（标准的 Kalman 滤波器步骤）
    x = F1 @ x
    P = F2 @ P @ F2.T + Q
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    
    Z = measurement  # 使用 DOA 和仰角作为测量
    y = Z - H @ x
    x = x + K @ y
    P = (np.eye(6) - K @ H) @ P

    # 估算目标的位置
    range_estimate = 100  # 这里是一个假设的估计值，可以通过 Bistatic Range 等方法获取
    target_x, target_y = doa_to_position(azimuth, elevation, range_estimate, x[4], x[5])  # 假设发射机/接收机坐标为 (x[4], x[5])
    
    updated_position = np.array([target_x, target_y])
    
    # 返回更新后的目标位置和状态
    newState = (x, P, F1, F2, Q, H, R, S)
    
    return updated_position, newState

# import matplotlib.pyplot as plt

def plot_target_trajectory(trajectory, title="Target Tracking"):
    """
    Plot the target trajectory on a 2D plot.
    
    :param trajectory: List of target positions (x, y)
    :param title: Title of the plot
    """
    x_vals = [pos[0] for pos in trajectory]
    y_vals = [pos[1] for pos in trajectory]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('X Position (km)')
    plt.ylabel('Y Position (km)')
    plt.grid(True)
    plt.show()

# 动态计算发射机和接收机的坐标，假设它们的位置随着时间变化
def dynamic_position_update(prev_x_tx, prev_y_tx, prev_x_rx, prev_y_rx, time_step):
    """
    假设发射机和接收机位置随着时间变化
    :param prev_x_tx: 发射机上一步的位置x
    :param prev_y_tx: 发射机上一步的位置y
    :param prev_x_rx: 接收机上一步的位置x
    :param prev_y_rx: 接收机上一步的位置y
    :param time_step: 当前时间步长，用于更新发射机和接收机位置
    :return: 更新后的发射机和接收机位置
    """
    # 假设发射机和接收机沿着x轴移动，具体移动模型可以根据实际情况调整
    updated_x_tx = prev_x_tx + 1 * time_step  # 发射机位置每秒移动1单位
    updated_y_tx = prev_y_tx  # 假设y轴不动
    updated_x_rx = prev_x_rx + 0.5 * time_step  # 接收机位置每秒移动0.5单位
    updated_y_rx = prev_y_rx  # 假设y轴不动

    return updated_x_tx, updated_y_tx, updated_x_rx, updated_y_rx

def doa_to_position(azimuth, elevation, range_estimate, rx_x, rx_y):
    """
    通过 DOA 和仰角计算目标的位置
    :param azimuth: 目标的方位角（度）
    :param elevation: 目标的仰角（度）
    :param range_estimate: 目标到接收机阵列的距离估算
    :param rx_x: 接收机阵列的 x 坐标
    :param rx_y: 接收机阵列的 y 坐标
    :return: 目标的笛卡尔坐标 (x, y)
    """
    # 将角度转换为弧度
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)

    # 计算目标的 x 和 y 坐标
    target_x = rx_x + range_estimate * np.cos(elevation) * np.cos(azimuth)
    target_y = rx_y + range_estimate * np.cos(elevation) * np.sin(azimuth)

    return target_x, target_y
