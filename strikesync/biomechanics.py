import numpy as np
from fastdtw import fastdtw


def calculate_angle(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def calculate_trunk_lean(mid_shoulder, mid_hip) -> float:
    vertical_vector = np.array([0, -1])
    trunk_vector = np.array([mid_shoulder[0] - mid_hip[0], mid_shoulder[1] - mid_hip[1]])
    cosine_angle = np.dot(vertical_vector, trunk_vector) / (
        np.linalg.norm(vertical_vector) * np.linalg.norm(trunk_vector) + 1e-7
    )
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def calculate_hip_shoulder_separation(l_sh, r_sh, l_hip, r_hip) -> float:
    sh_vec = np.array([l_sh[0] - r_sh[0], l_sh[2] - r_sh[2]])
    hip_vec = np.array([l_hip[0] - r_hip[0], l_hip[2] - r_hip[2]])
    cosine_angle = np.dot(sh_vec, hip_vec) / (np.linalg.norm(sh_vec) * np.linalg.norm(hip_vec) + 1e-7)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def score_segment(user_seg, exp_seg, baseline_seg) -> float:
    clean_user = [v for v in user_seg if v > 0]
    if len(clean_user) < 2:
        return 60.0

    target_raw = [v for v in exp_seg if v > 0] if len(exp_seg) >= 2 else baseline_seg
    if len(target_raw) < 2:
        return 60.0

    x_old, x_new = np.linspace(0, 1, len(target_raw)), np.linspace(0, 1, len(clean_user))
    target_resampled = np.interp(x_new, x_old, target_raw)

    u_arr, t_arr = np.array(clean_user, dtype=float), np.array(target_resampled, dtype=float)
    u_centered, t_centered = u_arr - np.mean(u_arr), t_arr - np.mean(t_arr)

    dist, _ = fastdtw(u_centered, t_centered, dist=1)
    avg_error = dist / len(clean_user)

    score = 100.0 * np.exp(-avg_error / 15.0)
    return round(max(15.0, min(98.5, score)), 1)
