import json
import time
from datetime import date
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from fastdtw import fastdtw
from sklearn.ensemble import IsolationForest

from strikesync.biomechanics import (
    calculate_angle,
    calculate_hip_shoulder_separation,
    calculate_trunk_lean,
    score_segment,
)
from strikesync.database import (
    add_history,
    add_user,
    get_history,
    get_profile,
    get_user_for_recovery,
    hash_password,
    init_db,
    login_user,
    update_password,
    update_profile,
)
from strikesync.paths import (
    BASELINES_PATH,
    EXPERTS_DIR,
    SIDEBAR_IMAGE,
    STREAMLIT_DIR,
    SYNCED_VIDEO,
    TEMP_USER_VIDEO,
    ensure_runtime_dirs,
)

st.set_page_config(page_title="StrikeSync AI", layout="wide")

ensure_runtime_dirs()
init_db()


def set_theme(theme_name: str) -> None:
    STREAMLIT_DIR.mkdir(parents=True, exist_ok=True)
    base_theme = "light" if theme_name == "Light Mode" else "dark"
    with open(STREAMLIT_DIR / "config.toml", "w") as f:
        f.write(f"[theme]\nbase='{base_theme}'\n")


def list_expert_videos() -> list[Path]:
    if not EXPERTS_DIR.exists():
        return []
    return sorted(
        p
        for p in EXPERTS_DIR.iterdir()
        if p.suffix.lower() in {".mp4", ".mov", ".avi"} and p.is_file()
    )


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "recovery_step" not in st.session_state:
    st.session_state["recovery_step"] = 0
if "recovery_user" not in st.session_state:
    st.session_state["recovery_user"] = None
if "font_size" not in st.session_state:
    st.session_state["font_size"] = "Medium"
if "ui_theme" not in st.session_state:
    st.session_state["ui_theme"] = "Dark Mode"

font_sizes = {"Small": "13px", "Medium": "16px", "Large": "20px"}
st.markdown(
    f"<style>html {{ font-size: {font_sizes.get(st.session_state['font_size'])} !important; }}</style>",
    unsafe_allow_html=True,
)

# ==========================================
# VIEW 1: AUTHENTICATION GATEWAY
# ==========================================
if not st.session_state["logged_in"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🏸 StrikeSync</h1>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='text-align: center; color: gray;'>AI Biomechanics Coaching Platform</h4>",
            unsafe_allow_html=True,
        )
        st.write("")

        auth_tab1, auth_tab2, auth_tab3 = st.tabs(["🔒 Login", "📝 Sign Up", "🔑 Forgot Password"])
        with auth_tab1:
            with st.form("login_form"):
                l_user, l_pass = st.text_input("Username"), st.text_input("Password", type="password")
                if st.form_submit_button("Login", use_container_width=True):
                    if user_data := login_user(l_user, l_pass):
                        st.session_state.update({"logged_in": True, "username": user_data[0]})
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please try again.")

        with auth_tab2:
            with st.form("signup_form"):
                s_user, s_pass = st.text_input("Choose Username *"), st.text_input(
                    "Choose Password *", type="password"
                )
                st.markdown("---")
                st.write("**Account Recovery Setup**")
                s_sec_q = st.selectbox(
                    "Select a security question: *",
                    [
                        "",
                        "What was the name of your first pet?",
                        "What is your mother's maiden name?",
                        "In what city were you born?",
                    ],
                )
                s_sec_a = st.text_input("Answer *")
                if st.form_submit_button("Create Account", use_container_width=True):
                    if not s_user or not s_pass:
                        st.error("Username and Password are required.")
                    elif not s_sec_q or not s_sec_a:
                        st.error("You must set a Security Question and Answer.")
                    elif add_user(s_user, s_pass, s_sec_q, s_sec_a):
                        st.success("Account created! You can now log in.")
                    else:
                        st.error("Username already exists in the database.")

        with auth_tab3:
            if st.session_state["recovery_step"] == 0:
                rec_id = st.text_input("Username", key="rec_id")
                if st.button("Search Account"):
                    if user_data := get_user_for_recovery(rec_id):
                        if user_data[1]:
                            st.session_state.update({"recovery_user": user_data, "recovery_step": 1})
                            st.rerun()
                    else:
                        st.error("Account not found or no security question was configured.")
            elif st.session_state["recovery_step"] == 1:
                u_name, u_sec_q, u_sec_a = st.session_state["recovery_user"]
                st.success(f"Account Located: **{u_name}**")
                ans = st.text_input(u_sec_q)
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("Verify Answer", use_container_width=True):
                        if hash_password(ans.strip().lower()) == u_sec_a:
                            st.session_state["recovery_step"] = 2
                            st.rerun()
                        else:
                            st.error("Incorrect answer. Please try again.")
                with col_btn2:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state["recovery_step"] = 0
                        st.rerun()
            elif st.session_state["recovery_step"] == 2:
                new_pass = st.text_input("Enter your New Password", type="password")
                if st.button("Reset Password & Login", use_container_width=True):
                    if new_pass:
                        update_password(st.session_state["recovery_user"][0], new_pass)
                        st.success("Password reset successfully! Logging you in...")
                        time.sleep(1.5)
                        st.session_state.update(
                            {
                                "recovery_step": 0,
                                "logged_in": True,
                                "username": st.session_state["recovery_user"][0],
                            }
                        )
                        st.rerun()
                    else:
                        st.error("Password cannot be blank.")

# ==========================================
# VIEW 2: MAIN PLATFORM (GATED)
# ==========================================
else:
    st.sidebar.title("🏸 StrikeSync")
    st.sidebar.write(f"Welcome back, **{st.session_state['username']}**")

    if SIDEBAR_IMAGE.exists():
        st.sidebar.image(str(SIDEBAR_IMAGE), use_container_width=True)

    nav_selection = st.sidebar.radio("Navigation", ["Dashboard", "My Profile", "History", "Settings"])
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.update({"logged_in": False, "username": "", "recovery_step": 0})
        st.rerun()

    if nav_selection == "Dashboard":
        st.title("Full-Body Biomechanics Engine")
        st.write("Upload your smash video. The AI tracks the entire kinetic chain with automatic phase segmentation.")

        try:
            with open(BASELINES_PATH, "r") as f:
                expert_db = json.load(f)
                json_expert_angles = expert_db["expert_average"]["elbow_angles"]
        except FileNotFoundError:
            json_expert_angles = [75, 80, 85, 90, 110, 135, 165, 175, 160, 120, 80, 45, 30]

        with st.expander("📊 Scoring Methodology & Validation Benchmarks"):
            st.markdown(
                "StrikeSync grades your kinetic chain using Dynamic Time Warping (DTW) to compute the distance between your joint trajectory and the professional baseline, adjusting for temporal shifting and execution speed."
            )
            st.latex(r"Match \% = \max\left(0, 100 - \frac{DTW(User_{traj}, Expert_{traj})}{N} \right)")
            st.markdown(
                """
            | Swing Quality | Score Range | Diagnostic Indicator |
            |---|---|---|
            | **Elite / Pro** | 85% - 100% | Near-perfect kinetic alignment, explosive physical ratio. |
            | **Good / Club** | 70% - 84% | Solid mechanics with minor temporal or angular deviations. |
            | **Average** | 50% - 69% | Noticeable form breaks, restricted extension, or poor leverage. |
            | **Poor** | < 50% | Dangerous mechanics posing high injury risk to rotator cuff/elbow. |
            """
            )

        col_vid1, col_vid2 = st.columns(2)
        with col_vid1:
            user_handedness = st.selectbox("Your Playing Hand", ["Right-Handed", "Left-Handed"])

            available_experts = list_expert_videos()
            expert_labels = [p.name for p in available_experts] + ["Baseline Average (JSON)"]
            expert_selection = st.selectbox("Select Expert Reference", expert_labels)

            user_file = st.file_uploader("Upload Your Smash (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
        with col_vid2:
            st.write("### 🧠 Phase-Segmented Analysis")
            st.info("Automatically segmenting into Preparation, Backswing, Jump, Impact, and Follow-Through.")

        if user_file is not None:
            with open(TEMP_USER_VIDEO, "wb") as f:
                f.write(user_file.getbuffer())

            if expert_selection == "Baseline Average (JSON)":
                target_expert_file = None
                has_expert = False
            else:
                target_expert_file = EXPERTS_DIR / expert_selection
                has_expert = target_expert_file.exists()

            mp_pose = mp.solutions.pose
            mp_draw = mp.solutions.drawing_utils
            joint_style = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=6, circle_radius=8)
            bone_style = mp_draw.DrawingSpec(color=(255, 0, 255), thickness=6, circle_radius=2)

            cap_user = cv2.VideoCapture(str(TEMP_USER_VIDEO))
            ret, first_frame = cap_user.read()

            if not ret:
                st.error("Failed to read user video.")
            else:
                raw_height, raw_width, _ = first_frame.shape
                width, height = int(raw_width) // 2 * 2, int(raw_height) // 2 * 2

                user_frames, user_data = [], {
                    "elbow": [],
                    "wrist_y": [],
                    "shoulder_abd": [],
                    "trunk_lean": [],
                    "separation": [],
                    "knee": [],
                    "hip_y": [],
                    "ank_y": [],
                    "body_len": [],
                }
                skew_frame_count = 0

                progress_bar = st.progress(0)
                st.write("Extracting Biomechanics...")

                u_sh, u_el, u_wr = (12, 14, 16) if user_handedness == "Right-Handed" else (11, 13, 15)
                u_hip, u_knee, u_ank = (24, 26, 28) if user_handedness == "Right-Handed" else (23, 25, 27)
                l_sh, r_sh, l_hip, r_hip = 11, 12, 23, 24

                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    while True:
                        ret_u, frame_u = cap_user.read()
                        if not ret_u:
                            break
                        frame_u = cv2.resize(frame_u, (width, height))
                        res_u = pose.process(cv2.cvtColor(frame_u, cv2.COLOR_BGR2RGB))

                        metrics = {k: height if "y" in k else 0 for k in user_data}

                        if res_u.pose_landmarks:
                            mp_draw.draw_landmarks(
                                frame_u, res_u.pose_landmarks, mp_pose.POSE_CONNECTIONS, joint_style, bone_style
                            )
                            lm = res_u.pose_landmarks.landmark

                            if abs(lm[11].x - lm[12].x) > 0.15 and abs(lm[11].z - lm[12].z) > 0.15:
                                skew_frame_count += 1

                            sh, el, wr = (
                                np.array([lm[u_sh].x * width, lm[u_sh].y * height]),
                                np.array([lm[u_el].x * width, lm[u_el].y * height]),
                                np.array([lm[u_wr].x * width, lm[u_wr].y * height]),
                            )
                            hip, knee, ank = (
                                np.array([lm[u_hip].x * width, lm[u_hip].y * height]),
                                np.array([lm[u_knee].x * width, lm[u_knee].y * height]),
                                np.array([lm[u_ank].x * width, lm[u_ank].y * height]),
                            )
                            mid_sh, mid_hp = (
                                np.array(
                                    [
                                        (lm[l_sh].x + lm[r_sh].x) / 2 * width,
                                        (lm[l_sh].y + lm[r_sh].y) / 2 * height,
                                    ]
                                ),
                                np.array(
                                    [
                                        (lm[l_hip].x + lm[r_hip].x) / 2 * width,
                                        (lm[l_hip].y + lm[r_hip].y) / 2 * height,
                                    ]
                                ),
                            )

                            l_sh_3d, r_sh_3d = [lm[l_sh].x, lm[l_sh].y, lm[l_sh].z], [
                                lm[r_sh].x,
                                lm[r_sh].y,
                                lm[r_sh].z,
                            ]
                            l_hp_3d, r_hp_3d = [lm[l_hip].x, lm[l_hip].y, lm[l_hip].z], [
                                lm[r_hip].x,
                                lm[r_hip].y,
                                lm[r_hip].z,
                            ]

                            metrics.update(
                                {
                                    "elbow": int(calculate_angle(sh, el, wr)),
                                    "wrist_y": int(wr[1]),
                                    "shoulder_abd": int(calculate_angle(hip, sh, el)),
                                    "trunk_lean": int(calculate_trunk_lean(mid_sh, mid_hp)),
                                    "separation": int(
                                        calculate_hip_shoulder_separation(l_sh_3d, r_sh_3d, l_hp_3d, r_hp_3d)
                                    ),
                                    "knee": int(calculate_angle(hip, knee, ank)),
                                    "hip_y": int(mid_hp[1]),
                                    "ank_y": int(ank[1]),
                                    "body_len": abs(int(sh[1]) - int(ank[1])),
                                }
                            )
                            cv2.putText(
                                frame_u,
                                f'Elb: {metrics["elbow"]}',
                                (int(el[0]) + 10, int(el[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                            )

                        for k, v in metrics.items():
                            user_data[k].append(v)
                        user_frames.append(frame_u)
                cap_user.release()

                if skew_frame_count > (len(user_frames) * 0.20):
                    st.warning(
                        "⚠️ **Camera Angle Skew Detected:** The 2D tracking engine detected significant diagonal depth variance across your shoulders. For highly accurate results, please ensure the camera is positioned directly to your side or squarely in front of you."
                    )

                for key in user_data:
                    arr = np.array(user_data[key], dtype=float)
                    zeros = arr == 0
                    if np.any(zeros) and not np.all(zeros):
                        idx = np.arange(len(arr))
                        arr[zeros] = np.interp(idx[zeros], idx[~zeros], arr[~zeros])
                        user_data[key] = arr.tolist()

                expert_frames, expert_angles, expert_wrist_y = [], [], []
                if has_expert:
                    st.write(f"Aligning with Expert Database ({expert_selection})...")
                    cap_expert = cv2.VideoCapture(str(target_expert_file))

                    expert_name = expert_selection.lower()
                    if "left" in expert_name:
                        e_sh, e_el, e_wr = 11, 13, 15
                    elif "right" in expert_name:
                        e_sh, e_el, e_wr = 12, 14, 16
                    else:
                        e_sh, e_el, e_wr = (12, 14, 16) if user_handedness == "Right-Handed" else (11, 13, 15)

                    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_exp:
                        while True:
                            ret_e, frame_e = cap_expert.read()
                            if not ret_e:
                                break
                            frame_e = cv2.resize(frame_e, (width, height))
                            res_e = pose_exp.process(cv2.cvtColor(frame_e, cv2.COLOR_BGR2RGB))

                            e_elbow_angle = 0
                            e_w_y = height
                            if res_e.pose_landmarks:
                                mp_draw.draw_landmarks(
                                    frame_e, res_e.pose_landmarks, mp_pose.POSE_CONNECTIONS, joint_style, bone_style
                                )
                                lm = res_e.pose_landmarks.landmark
                                sh_c, el_c, wr_c = (
                                    np.array([lm[e_sh].x * width, lm[e_sh].y * height]),
                                    np.array([lm[e_el].x * width, lm[e_el].y * height]),
                                    np.array([lm[e_wr].x * width, lm[e_wr].y * height]),
                                )
                                e_elbow_angle = int(calculate_angle(sh_c, el_c, wr_c))
                                e_w_y = int(wr_c[1])

                            expert_angles.append(e_elbow_angle)
                            expert_wrist_y.append(e_w_y)
                            expert_frames.append(frame_e)
                    cap_expert.release()

                    arr_e = np.array(expert_angles, dtype=float)
                    if np.any(arr_e == 0) and not np.all(arr_e == 0):
                        idx = np.arange(len(arr_e))
                        arr_e[arr_e == 0] = np.interp(idx[arr_e == 0], idx[arr_e != 0], arr_e[arr_e != 0])
                        expert_angles = arr_e.tolist()

                    arr_ey = np.array(expert_wrist_y, dtype=float)
                    if np.any(arr_ey == height) and not np.all(arr_ey == height):
                        idx = np.arange(len(arr_ey))
                        arr_ey[arr_ey == height] = np.interp(
                            idx[arr_ey == height], idx[arr_ey != height], arr_ey[arr_ey != height]
                        )
                        expert_wrist_y = arr_ey.tolist()

                st.write("Synchronizing Timelines & Generating Local Video...")
                can_hconcat = has_expert and len(user_data["elbow"]) > 0 and len(expert_angles) > 0
                out_width = width * 2 if can_hconcat else width

                output_path = str(SYNCED_VIDEO.resolve())
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                out_vid = cv2.VideoWriter(output_path, fourcc, 30, (out_width, height))

                if can_hconcat:
                    dist, path = fastdtw(user_data["elbow"], expert_angles, dist=1)
                    match_percentage = max(0, 100 - (dist / len(path)))

                    u_impact = np.argmin(user_data["wrist_y"]) if len(user_data["wrist_y"]) > 0 else len(user_frames) // 2
                    e_impact = np.argmin(expert_wrist_y) if len(expert_wrist_y) > 0 else len(expert_frames) // 2

                    u_impact = max(1, min(u_impact, len(user_frames) - 2))
                    e_impact = max(1, min(e_impact, len(expert_frames) - 2))

                    for u_idx in range(len(user_frames)):
                        if u_idx <= u_impact:
                            e_idx = int(u_idx * (e_impact / u_impact))
                        else:
                            u_rem = len(user_frames) - 1 - u_impact
                            e_rem = len(expert_frames) - 1 - e_impact
                            ratio = e_rem / u_rem if u_rem > 0 else 1
                            e_idx = e_impact + int((u_idx - u_impact) * ratio)

                        e_idx = min(max(0, e_idx), len(expert_frames) - 1)
                        out_vid.write(cv2.hconcat([user_frames[u_idx], expert_frames[e_idx]]))
                else:
                    for f_u in user_frames:
                        out_vid.write(f_u)
                    if user_data["elbow"]:
                        match_percentage = max(
                            0,
                            100 - (fastdtw(user_data["elbow"], json_expert_angles, dist=1)[0] / len(user_data["elbow"])),
                        )
                    else:
                        match_percentage = 0.0

                out_vid.release()
                progress_bar.progress(1.0)
                add_history(st.session_state["username"], round(match_percentage, 1))

                total_f = len(user_data["elbow"])
                if total_f >= 10:
                    impact_i = np.argmin(user_data["wrist_y"])
                    pre_impact_elbows, pre_impact_hips = (
                        user_data["elbow"][: impact_i + 1] if impact_i > 0 else user_data["elbow"],
                        user_data["hip_y"][: impact_i + 1] if impact_i > 0 else user_data["hip_y"],
                    )
                    backswing_i, jump_i = (
                        np.argmin(pre_impact_elbows) if len(pre_impact_elbows) > 0 else int(impact_i * 0.3),
                        np.argmin(pre_impact_hips) if len(pre_impact_hips) > 0 else int(impact_i * 0.6),
                    )

                    prep_end = max(2, min(backswing_i, int(total_f * 0.2)))
                    back_end = max(prep_end + 2, min(jump_i, int(total_f * 0.45)))
                    jump_end = max(back_end + 2, min(impact_i, int(total_f * 0.7)))
                    impact_end = min(total_f - 2, max(jump_end + 2, impact_i + 3))

                    u_prep, u_back, u_imp, u_foll = (
                        user_data["elbow"][:prep_end],
                        user_data["elbow"][prep_end:back_end],
                        user_data["elbow"][jump_end:impact_end],
                        user_data["elbow"][impact_end:],
                    )

                    if has_expert and len(expert_angles) >= 10:
                        e_len = len(expert_angles)
                        e_prep, e_back, e_imp, e_foll = (
                            expert_angles[: int(e_len * (prep_end / total_f))],
                            expert_angles[int(e_len * (prep_end / total_f)) : int(e_len * (back_end / total_f))],
                            expert_angles[int(e_len * (jump_end / total_f)) : int(e_len * (impact_end / total_f))],
                            expert_angles[int(e_len * (impact_end / total_f)) :],
                        )
                    else:
                        e_prep = e_back = e_imp = e_foll = []

                    b_len = len(json_expert_angles)
                    b_prep, b_back, b_imp, b_foll = (
                        json_expert_angles[: max(2, int(b_len * 0.2))],
                        json_expert_angles[int(b_len * 0.2) : max(3, int(b_len * 0.45))],
                        json_expert_angles[int(b_len * 0.7) : max(5, int(b_len * 0.85))],
                        json_expert_angles[int(b_len * 0.85) :],
                    )

                    score_prep, score_back, score_imp, score_foll = (
                        score_segment(u_prep, e_prep, b_prep),
                        score_segment(u_back, e_back, b_back),
                        score_segment(u_imp, e_imp, b_imp),
                        score_segment(u_foll, e_foll, b_foll),
                    )

                    standing_ank_y, peak_ank_y = (
                        np.max(user_data["ank_y"]) if user_data["ank_y"] else height,
                        np.min(user_data["ank_y"]) if user_data["ank_y"] else height,
                    )
                    jump_pixels = max(0, standing_ank_y - peak_ank_y)
                    u_body_px = np.median(user_data["body_len"]) if user_data["body_len"] else height
                    u_jump_ratio = jump_pixels / u_body_px if u_body_px > 0 else 0
                    score_jump = round(min(100.0, u_jump_ratio * 1.5 * 100.0), 1)

                else:
                    score_prep = score_back = score_jump = score_imp = score_foll = round(match_percentage, 1)
                    jump_pixels = 0

                st.success("✅ Synchronization & Processing Complete")
                st.info(
                    f"📁 **File saved locally at:**\n`{output_path}`\n\n_Play this file directly in QuickTime for high-framerate side-by-side judging._"
                )
                st.markdown("---")

                st.markdown("### ⏱️ Phase Breakdown & Targeted Feedback")
                col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
                col_p1.metric("1. Preparation", f"{score_prep:.1f}%")
                col_p2.metric("2. Backswing", f"{score_back:.1f}%")
                col_p3.metric("3. Jump / Takeoff", f"{score_jump:.1f}%")
                col_p4.metric("4. Impact", f"{score_imp:.1f}%")
                col_p5.metric("5. Follow-Through", f"{score_foll:.1f}%")

                st.write("")
                feedback = []
                if score_prep < 70:
                    feedback.append("⚠️ **Preparation:** Racket preparation is delayed. Raise your arm earlier.")
                else:
                    feedback.append("✅ **Preparation:** Strong ready stance and racket position.")
                if score_back < 70:
                    feedback.append(
                        "⚠️ **Backswing:** Your backswing is too short or restricted. Pull the elbow further back."
                    )
                else:
                    feedback.append("✅ **Backswing:** Deep backswing arc delivering optimal store of elastic energy.")
                if score_jump < 70:
                    feedback.append(
                        "⚠️ **Jump / Takeoff:** Explosive vertical output is low relative to body height. Flex knees deeper before loading."
                    )
                else:
                    feedback.append("✅ **Jump / Takeoff:** Excellent vertical thrust and physical displacement mechanics.")
                if score_imp < 70:
                    feedback.append(
                        "⚠️ **Impact:** Arm extension is premature or bent at contact. Full kinetic chain extension is missing."
                    )
                else:
                    feedback.append("✅ **Impact:** Optimal peak extension hitting the shuttle high in front of your body.")
                if score_foll < 70:
                    feedback.append(
                        "⚠️ **Follow-Through:** Swing cuts off sharply. Allow full rotational deceleration across your body."
                    )
                else:
                    feedback.append("✅ **Follow-Through:** Smooth kinetic deceleration protecting shoulder joints.")

                for line in feedback:
                    if line.startswith("⚠️"):
                        st.warning(line)
                    else:
                        st.success(line)

                st.markdown("---")
                st.markdown("### 📊 Kinetic Chain Metrics (At Impact)")
                impact_idx = np.argmin(user_data["wrist_y"]) if user_data["wrist_y"] else 0
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Elbow Extension", f"{int(user_data['elbow'][impact_idx])}°")
                col_m2.metric("Shoulder Abduction", f"{int(user_data['shoulder_abd'][impact_idx])}°")
                col_m3.metric("Trunk Lean", f"{int(user_data['trunk_lean'][impact_idx])}°")
                col_m4.metric("Hip-Shoulder Separation", f"{int(user_data['separation'][impact_idx])}°")

                st.markdown("### 🚀 Jump Mechanics")
                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                takeoff_idx = np.argmin(user_data["hip_y"][: impact_idx + 1]) if impact_idx > 0 else 0
                landing_idx = (
                    np.argmax(user_data["hip_y"][impact_idx:]) + impact_idx
                    if impact_idx < len(user_data["hip_y"])
                    else -1
                )

                col_b1.metric("Max Jump Height (Px)", f"{int(jump_pixels)} px")
                col_b2.metric("Knee Flexion (Takeoff)", f"{int(user_data['knee'][takeoff_idx])}°")
                col_b3.metric("Knee Flexion (Landing)", f"{int(user_data['knee'][landing_idx])}°")
                col_b4.metric("Overall Match", f"{match_percentage:.1f}%")

                if len(user_data["elbow"]) > 10:
                    clf = IsolationForest(contamination=0.05, random_state=42)
                    if np.sum(clf.fit_predict(np.array(user_data["elbow"]).reshape(-1, 1)) == -1) > 3:
                        st.error("🚨 Anomalous Swing Arc Detected")
                    else:
                        st.success("✅ Optimal Mechanical Flow")

    elif nav_selection == "My Profile":
        st.title("Athlete Profile")
        st.write("Customize your physical metrics to improve AI coaching accuracy in future updates.")
        current_data = get_profile(st.session_state["username"])
        curr_dob, curr_weight, curr_height = (
            current_data[0] if current_data and current_data[0] else None,
            current_data[1] if current_data and current_data[1] else 0.0,
            current_data[2] if current_data and current_data[2] else 0.0,
        )

        today = date.today()
        try:
            min_dob, max_dob = today.replace(year=today.year - 100), today.replace(year=today.year - 5)
        except ValueError:
            min_dob, max_dob = today.replace(year=today.year - 100, day=28), today.replace(year=today.year - 5, day=28)
        default_dob = max(min_dob, min(pd.to_datetime(curr_dob).date() if curr_dob else max_dob, max_dob))

        with st.form("profile_form"):
            dob, weight, height = (
                st.date_input("Date of Birth", value=default_dob, min_value=min_dob, max_value=max_dob),
                st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, value=curr_weight, step=0.5),
                st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=curr_height, step=1.0),
            )
            if st.form_submit_button("Save Profile"):
                update_profile(st.session_state["username"], str(dob), weight, height)
                st.success("Profile successfully updated!")

    elif nav_selection == "History":
        st.title("Training Progression")
        if not (df := get_history(st.session_state["username"])).empty:
            st.line_chart(df.set_index("Date")["Match (%)"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No smash history found. Process a video in the Dashboard to track your progress.")
    elif nav_selection == "Settings":
        st.title("Application Settings")
        st.write("Customize your StrikeSync environment.")

        theme_idx = 0 if st.session_state["ui_theme"] == "Dark Mode" else 1
        font_idx = ["Small", "Medium", "Large"].index(st.session_state["font_size"])

        theme_choice = st.radio("UI Theme", ["Dark Mode", "Light Mode"], index=theme_idx)
        font_choice = st.selectbox("Global Font Size", ["Small", "Medium", "Large"], index=font_idx)

        if st.button("Apply Settings"):
            st.session_state["ui_theme"] = theme_choice
            st.session_state["font_size"] = font_choice
            set_theme(theme_choice)
            time.sleep(0.5)
            st.rerun()
