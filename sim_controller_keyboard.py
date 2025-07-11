import time
import numpy as np
import cv2
import signal
from pynput import keyboard
from coppeliasim import Coppeliasim

# === 제어 상수 ===
STEER_DELTA = 0.05
SPEED_DELTA = 0.05
MAX_STEER = 0.5
MAX_SPEED = 0.5
SEND_INTERVAL = 0.05

# === 상태 변수 ===
steer = 0.0
speed = 0.0
run_flag = True

# === CoppeliaSim 연결 ===
sim = Coppeliasim()
sim.read_data.img_flag = True
out = None


# 2.637
# === 클램프 함수 ===
def clamp(val, max_val):
    return max(min(val, max_val), -max_val)


# === 안전 종료 핸들러 ===
def signal_handler(sig, frame):
    global run_flag
    print("\n[SIM] Ctrl+C detected. Exiting safely...")
    run_flag = False
    sim.run_flag = False


signal.signal(signal.SIGINT, signal_handler)


# === 키보드 입력 처리 ===
def on_press(key):
    global steer, speed, run_flag
    try:
        if key.char == "w":
            speed = clamp(speed + SPEED_DELTA, MAX_SPEED)
        elif key.char == "s":
            speed = clamp(speed - SPEED_DELTA, MAX_SPEED)
        elif key.char == "d":
            steer = clamp(steer + STEER_DELTA, MAX_STEER)
        elif key.char == "a":
            steer = clamp(steer - STEER_DELTA, MAX_STEER)
    except AttributeError:
        if key == keyboard.Key.space:
            print("[SIM] 정지 (space bar)")
            steer = 0.0
            speed = 0.0
        elif key == keyboard.Key.esc:
            run_flag = False
            sim.run_flag = False
            return False


# === 콜백 루프 ===
def step_sim(read_data, control_data):
    global out, steer, speed

    if not run_flag:
        return

    # 영상 저장 초기화
    if out is None and read_data.img_flag and read_data.img is not None:
        h, w, _ = read_data.img.shape
        out = cv2.VideoWriter(
            "sim_keyboard_output_2.avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            10,
            (w, h),
        )

    # 조향 처리
    curr = read_data.joints
    if abs(steer) > 1e-3:
        left_speed = steer
        right_speed = -steer
    else:
        left_speed = speed
        right_speed = speed

    control_data.wheels_position = (
        curr[0] + left_speed,
        curr[1] + right_speed,
        curr[2] + left_speed,
        curr[3] + right_speed,
    )
    print(f"L: {left_speed} | R: {right_speed}")

    # 카메라 영상 기록
    if read_data.img_flag and read_data.img is not None:
        bgr_img = cv2.cvtColor(read_data.img, cv2.COLOR_RGB2BGR)
        out.write(bgr_img)


# === 메인 ===
if __name__ == "__main__":
    print("[SIM] Keyboard control: w/s/a/d = 방향, space = 정지, esc or Ctrl+C = 종료")
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    sim.run(step_sim)
    if out is not None:
        out.release()
    print("[SIM] 종료 완료.")
