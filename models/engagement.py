import pandas as pd
import numpy as np
import os

def find_common_viewpoint(csv_file):
    data = pd.read_csv(csv_file)
    
    zones = ["left", "center", "right"]
    zone_median_pose = {}

    for zone in zones:
        zone_data = data[data['zone'] == zone]
        if not zone_data.empty:
            median_pitch = zone_data['pose.pitch'].median()
            median_yaw = zone_data['pose.yaw'].median()
            median_roll = zone_data['pose.roll'].median()
            zone_median_pose[zone] = {
                "median_pitch": median_pitch,
                "median_yaw": median_yaw,
                "median_roll": median_roll,
            }
    
    return zone_median_pose

def calculate_engagement(csv_file):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"The specified file does not exist: {csv_file}")
    
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"Failed to read the CSV file: {csv_file}. Error: {e}")

    data['pose.pitch'] = pd.to_numeric(data['pose.pitch'], errors='coerce')
    data['pose.yaw'] = pd.to_numeric(data['pose.yaw'], errors='coerce')
    data['pose.roll'] = pd.to_numeric(data['pose.roll'], errors='coerce')
    data['confidence'] = pd.to_numeric(data['confidence'], errors='coerce')

    # ── Eye-state and gaze weights (replaces emotion weights) ───────────────
    eye_state_weights = {
        "open": 0,                # Fully attentive
        "partially_closed": -20,  # Drowsy / squinting
        "closed": -50,            # Eyes closed — likely disengaged
    }

    gaze_weights = {
        "center": 0,    # Looking forward — engaged
        "left": -10,    # Looking sideways — mildly distracted
        "right": -10,
        "away": -30,    # Looking away — disengaged
    }

    zone_median_pose = find_common_viewpoint(csv_file)
    
    engagement_scores = []
    
    for _, row in data.iterrows():
        face_id = row["face_id"]
        zone = row["zone"]
        eye_state = row.get("eye_state", "closed")
        gaze = row.get("gaze", "away")
        confidence = row["confidence"]
        eyes_detected = int(row.get("eyes_detected", 0))

        # ── Head-pose deviation score (always computed) ─────────────────────
        zone_pose = zone_median_pose.get(zone, {
            "median_pitch": 0, "median_yaw": 0, "median_roll": 0
        })

        pitch_deviation = abs(row["pose.pitch"] - zone_pose["median_pitch"])
        yaw_deviation = abs(row["pose.yaw"] - zone_pose["median_yaw"])
        roll_deviation = abs(row["pose.roll"] - zone_pose.get("median_roll", 0))

        if yaw_deviation > 90:
            yaw_deviation = 100
        if pitch_deviation > 100:
            pitch_deviation = 100
        if roll_deviation > 90:
            roll_deviation = 100

        max_deviation = 45
        yaw_score = max(0, 100 - (yaw_deviation / max_deviation) * 100)
        pitch_score = max(0, 100 - (pitch_deviation / max_deviation) * 100)
        roll_score = max(0, 100 - (roll_deviation / max_deviation) * 100)

        head_pose_score = (yaw_score * 0.7) + (pitch_score * 0.3)

        # ── Scoring branch ──────────────────────────────────────────────────
        if eyes_detected == 0:
            # Eyes NOT detected → score purely from head pose + roll correction
            # 85 % head direction  +  15 % roll stability
            total_engagement_score = (head_pose_score * 0.85) + (roll_score * 0.15)
            total_engagement_score = np.clip(total_engagement_score, 0, 100)
            eye_score = None          # signal that eyes were not available
        else:
            # Eyes detected → use normal eye-based attentiveness blend
            eye_weight = eye_state_weights.get(str(eye_state), -50)
            gaze_weight = gaze_weights.get(str(gaze), -30)
            eye_score = np.clip(100 + eye_weight + gaze_weight, 0, 100)

            # 60 % head pose  +  40 % eye attentiveness
            total_engagement_score = (head_pose_score * 0.6) + (eye_score * 0.4)
            total_engagement_score = np.clip(total_engagement_score, 0, 100)

        engagement_scores.append({
            "face_id": face_id,
            "zone": zone,
            "eyes_detected": eyes_detected,
            "eye_state": eye_state,
            "gaze": gaze,
            "confidence": confidence,
            "eye_score": eye_score,
            "pitch_deviation": pitch_deviation,
            "yaw_deviation": yaw_deviation,
            "roll_deviation": roll_deviation,
            "pitch_score": pitch_score,
            "yaw_score": yaw_score,
            "roll_score": roll_score,
            "head_pose_score": head_pose_score,
            "engagement_score": total_engagement_score
        })
    
    engagement_df = pd.DataFrame(engagement_scores)
    
    overall_engagement_score = engagement_df["engagement_score"].mean()

    return engagement_df, overall_engagement_score
