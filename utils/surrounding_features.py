# utils/surrounding_features.py
import numpy as np
import pandas as pd

def get_surrounding_features(frame_df, ego_row):
    if not isinstance(ego_row, pd.Series):
        return np.zeros(24)

    ego_id = ego_row['Vehicle_ID']
    ego_lane = ego_row['Lane_ID']
    ego_y = ego_row['Local_Y']
    ego_v = ego_row['v_Vel']
    ego_a = ego_row['v_Acc']

    neighbors = frame_df[frame_df['Vehicle_ID'] != ego_id]
    if neighbors.empty:
        return np.zeros(24)

    lane_ids = neighbors['Lane_ID'].values
    local_y = neighbors['Local_Y'].values
    v = neighbors['v_Vel'].values
    a = neighbors['v_Acc'].values

    rel_y = local_y - ego_y
    rel_v = v - ego_v
    rel_a = a - ego_a
    rel_lane = lane_ids - ego_lane

    masks = {
        'lead':         (rel_lane == 0)  & (rel_y > 0),
        'follow':       (rel_lane == 0)  & (rel_y < 0),
        'left_lead':    (rel_lane == -1) & (rel_y > 0),
        'left_follow':  (rel_lane == -1) & (rel_y < 0),
        'right_lead':   (rel_lane == 1)  & (rel_y > 0),
        'right_follow': (rel_lane == 1)  & (rel_y < 0),
    }

    def get_nearest(mask, fallback=None):
        if np.any(mask):
            rel_y_masked = rel_y[mask]
            rel_v_masked = rel_v[mask]
            rel_a_masked = rel_a[mask]
            rel_lane_masked = rel_lane[mask]
            idx = np.argmin(np.abs(rel_y_masked))
            return np.array([
                rel_y_masked[idx],
                rel_v_masked[idx],
                rel_a_masked[idx],
                rel_lane_masked[idx],
            ])
        elif fallback is not None and np.any(fallback):
            rel_y_masked = rel_y[fallback]
            rel_v_masked = rel_v[fallback]
            rel_a_masked = rel_a[fallback]
            rel_lane_masked = rel_lane[fallback]
            idx = np.argmin(np.abs(rel_y_masked))
            return np.array([
                rel_y_masked[idx],
                rel_v_masked[idx],
                rel_a_masked[idx],
                rel_lane_masked[idx],
            ])
        else:
            return np.zeros(4)

    features = np.concatenate([
        get_nearest(masks['lead'],         fallback=(rel_y > 0)),
        get_nearest(masks['follow'],       fallback=(rel_y < 0)),
        get_nearest(masks['left_lead'],    fallback=(rel_y > 0)),
        get_nearest(masks['left_follow'],  fallback=(rel_y < 0)),
        get_nearest(masks['right_lead'],   fallback=(rel_y > 0)),
        get_nearest(masks['right_follow'], fallback=(rel_y < 0)),
    ])

    return features
