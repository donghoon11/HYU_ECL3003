import time
import cv2
import numpy as np
import socket
from ultralytics import YOLO
from jetcam.csi_camera import CSICamera
from base_ctrl import BaseController
from multiprocessing import Process, Manager
import sys
import os

# 로그 저장 설정
log_path = "0614_patrol_save.log"
sys.stdout = open(log_path, "w", buffering=1)
sys.stderr = sys.stdout

# === 모델 로드 ===
road_model = YOLO("/home/ircv14/HYU-ECL3003/rover/models/goat_ver1_sNr.engine")

# === 실측 높이 설정 ===
REAL_HEIGHTS = {
    "yellow": 0.055,
    "blue": 0.055,
    "red": 0.055,
    "white": 0.055,
    "green": 0.055,
    "purple": 0.055,
}

# === FSM 상태 정의 ===
STATE_Y = "YELLOW_LOOP"
STATE_B = "BLUE_LOOP"
STATE_R = "RED_LOOP"
STATE_Y_W_B = "Y_W_B"
STATE_Y_W_R = "Y_W_R"
STATE_B_W_Y = "B_W_Y"
STATE_R_W_Y = "R_W_Y"
STATE_Y_G_B = "Y_G_B"
STATE_B_G_Y = "B_G_Y"
STATE_Y_P_R = "Y_P_R"
STATE_R_P_Y = "R_P_Y"
STATE_SEARCH = "SEARCH"
STATE_STOP = "STOP"

# === 설정 ===
SEARCH_DURATION = 1.0
FOCAL_LENGTH_PX = 1650
CONFIDENCE_THRESHOLD = 0.5
DOWNSAMPLE = 2
YOUR_IP = ""

def command_server(shared_command):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", 9999))
    s.listen(1)
    print("[COMMAND SERVER] Listening on port 9999...")
    while True:
        conn, addr = s.accept()
        with conn:
            data = conn.recv(1024).decode().strip().lower()
            if data:
                print(f"[COMMAND] Received: {data}")
                shared_command["value"] = data


def send_to_simulator(fsm_state, L, R):
    try:
        sim_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sim_sock.connect((YOUR_IP, 5007))  # 로컬PC_Ip, 포트번호
        msg = f"{fsm_state},{L:.3f},{R:.3f}"
        sim_sock.sendall(msg.encode("utf-8"))
        print(f"[SIM SYNC] Sent: {fsm_state}, L={L:.3f}, R={R:.3f}")
        sim_sock.close()
    except Exception as e:
        print(f"[SIM SYNC] Send failed: {e}")


class UGVAutoController:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200):
        self.base_ctrl = BaseController(port, baudrate)
        self.MAX_STEER = 0.95  # mean : 0.95
        self.MAX_SPEED = 0.4
        self.STEER_GAIN = 0.004

    def send_ctrl(self, steering, speed):
        steer_val = clip(steering, self.MAX_STEER)
        speed_val = clip(speed, self.MAX_SPEED)
        base_speed = abs(speed_val)
        left_ratio = 1.0 - steer_val
        right_ratio = 1.0 + steer_val
        L = clip(base_speed * left_ratio, self.MAX_SPEED)
        R = clip(base_speed * right_ratio, self.MAX_SPEED)
        if speed_val < 0:
            L, R = -L, -R
        self.base_ctrl.base_json_ctrl({"T": 1, "L": L, "R": R})
        return L, R


def clip(val, max_val):
    return max(min(val, max_val), -max_val)


def process_detections(results, model, frame):
    detected_classes = set()
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])
        if conf < CONFIDENCE_THRESHOLD:
            continue
        detected_classes.add(class_name)
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        color = {
            "yellow": (0, 255, 255),
            "blue": (255, 0, 0),
            "red": (0, 0, 255),
            "white": (255, 255, 255),
            "green": (0, 255, 0),
            "purple": (255, 0, 255),
        }.get(class_name, (100, 100, 100))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{class_name} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return detected_classes


def extract_center_points(results_line, model, target_class):
    points = []
    for box in results_line.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        if confidence >= CONFIDENCE_THRESHOLD and class_name == target_class:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            points.append(center)
    return points


def patrol_loop(shared_command):
    print("[FSM INIT] Waiting for start command (yellow/red/blue zone start)...")
    while True:
        cmd = shared_command["value"]
        if cmd in ["yellow zone start", "red zone start", "blue zone start"]:
            break
        time.sleep(0.1)

    fsm_state = {
        "yellow zone start": STATE_Y,
        "red zone start": STATE_R,
        "blue zone start": STATE_B,
    }[cmd]

    print(f"[FSM INIT] Starting FSM with state: {fsm_state}")
    last_valid_state = fsm_state
    fsm_timer_start = None
    total_loop_mode = False

    camera = CSICamera(
        capture_width=3280, capture_height=2464, downsample=DOWNSAMPLE, capture_fps=21
    )

    ugv = UGVAutoController()
    width, height = 3280, 2464
    out = cv2.VideoWriter(
        "0614_patrol_1.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height)
    )

    while True:
        print(f"[FSM] Current State: {fsm_state}")
        frame = camera.read()
        results = road_model.predict(frame, verbose=False)[0]
        detections = process_detections(results, road_model, frame)

        cmd = shared_command["value"]
        if cmd == "stop":
            fsm_state = STATE_STOP
            total_loop_mode = False
        elif cmd == "total loop":
            total_loop_mode = True
            shared_command["value"] = None
        elif cmd in [
            "yellow zone start",
            "red zone start",
            "blue zone start",
            "move to red zone",
            "move to blue zone",
            "move to yellow zone",
        ]:
            total_loop_mode = False

        if fsm_state == STATE_STOP:
            print("[FSM] STOP")
            ugv.send_ctrl(0.0, 0.0)
            break

        if fsm_state == STATE_Y:
            if not total_loop_mode:
                if cmd == "move to red zone" and "white" in detections:
                    fsm_state = STATE_Y_W_R
                    print("[FSM] Transition: Y → Y_W_R")
                    continue
                elif cmd == "move to blue zone" and "white" in detections:
                    fsm_state = STATE_Y_W_B
                    print("[FSM] Transition: Y → Y_W_B")
                    continue
            target_points = extract_center_points(results, road_model, "yellow")
            if target_points:
                last_valid_state = fsm_state

        elif fsm_state == STATE_Y_W_R:
            if "red" in detections:
                fsm_state = STATE_R
                print("[FSM] Transition: Y_W_R → R")
                continue
            target_points = extract_center_points(results, road_model, "white")
            if target_points:
                last_valid_state = fsm_state

        elif fsm_state == STATE_Y_W_B:
            if "blue" in detections:
                fsm_state = STATE_B
                print("[FSM] Transition: Y_W_B → B")
                continue
            target_points = extract_center_points(results, road_model, "white")
            if target_points:
                last_valid_state = fsm_state

        elif fsm_state == STATE_B:
            if (
                not total_loop_mode
                and cmd == "move to yellow zone"
                and "white" in detections
            ):
                fsm_state = STATE_B_W_Y
                print("[FSM] Transition: B → B_W_Y")
                continue
            target_points = extract_center_points(results, road_model, "blue")
            if target_points:
                last_valid_state = fsm_state

        elif fsm_state == STATE_B_W_Y:
            if "yellow" in detections:
                fsm_state = STATE_Y
                print("[FSM] Transition: B_W_Y → Y")
                continue
            target_points = extract_center_points(results, road_model, "white")
            if target_points:
                last_valid_state = fsm_state

        elif fsm_state == STATE_R:
            if (
                not total_loop_mode
                and cmd == "move to yellow zone"
                and "white" in detections
            ):
                fsm_state = STATE_R_W_Y
                print("[FSM] Transition: R → R_W_Y")
                continue
            target_points = extract_center_points(results, road_model, "red")
            if target_points:
                last_valid_state = fsm_state

        elif fsm_state == STATE_R_W_Y:
            if "yellow" in detections:
                fsm_state = STATE_Y
                print("[FSM] Transition: R_W_Y → Y")
                continue
            target_points = extract_center_points(results, road_model, "white")
            if target_points:
                last_valid_state = fsm_state

        # Total loop FSM
        if total_loop_mode:
            print("[FSM] Total loop mode active")
            if fsm_state == STATE_Y and "purple" in detections:
                fsm_state = STATE_Y_P_R
            elif fsm_state == STATE_R and "purple" in detections:
                fsm_state = STATE_R_P_Y
            elif fsm_state == STATE_Y and "green" in detections:
                fsm_state = STATE_Y_G_B
            elif fsm_state == STATE_B and "green" in detections:
                fsm_state = STATE_B_G_Y
            elif fsm_state == STATE_Y_P_R and "red" in detections:
                fsm_state = STATE_R
            elif fsm_state == STATE_R_P_Y and "yellow" in detections:
                fsm_state = STATE_Y
            elif fsm_state == STATE_Y_G_B and "blue" in detections:
                fsm_state = STATE_B
            elif fsm_state == STATE_B_G_Y and "yellow" in detections:
                fsm_state = STATE_Y

        if fsm_state == STATE_Y_P_R:
            target_points = extract_center_points(results, road_model, "purple")
        elif fsm_state == STATE_R_P_Y:
            target_points = extract_center_points(results, road_model, "purple")
        elif fsm_state == STATE_Y_G_B:
            target_points = extract_center_points(results, road_model, "green")
        elif fsm_state == STATE_B_G_Y:
            target_points = extract_center_points(results, road_model, "green")

        elif fsm_state == STATE_SEARCH:
            if time.time() - fsm_timer_start < SEARCH_DURATION:
                ugv.send_ctrl(0.0, -0.2)
                send_to_simulator(fsm_state, -0.2, -0.2)
                out.write(frame)
                time.sleep(0.1)
                continue
            else:
                print("[FSM] SEARCH COMPLETE")
                if last_valid_state and last_valid_state != STATE_SEARCH:
                    fsm_state = last_valid_state
                    print(f"[FSM] Returning to {fsm_state}")
                else:
                    print("[FSM] Invalid fallback. Resetting to YELLOW_LOOP")
                    fsm_state = STATE_Y
                continue

        # mean 조향 처리
        # if target_points:
        #     x_mean = int(np.mean([pt[0] for pt in target_points]))
        #     error = x_mean - (width // 2)
        #     steering = -error * ugv.STEER_GAIN
        #     L, R = ugv.send_ctrl(steering, ugv.MAX_SPEED)
        #     send_to_simulator(fsm_state, L, R)
        #     cv2.arrowedLine(
        #         frame,
        #         (width // 2, height - 30),
        #         (x_mean, height - 100),
        #         (0, 0, 255),
        #         2,
        #         tipLength=0.3,
        #     )
        #     print(f"[FSM: {fsm_state}] Steering: {steering:.3f}")
        # else:
        #     print(f"[FSM: {fsm_state}] No target point found → SEARCH")
        #     fsm_state = STATE_SEARCH
        #     fsm_timer_start = time.time()

        # bottom 조향 처리
        if target_points:
            target_points = sorted(target_points, key=lambda pt: pt[1])
            target = max(target_points, key=lambda pt: pt[1])
            error = target[0] - (width // 2)
            steering = -error * ugv.STEER_GAIN
            ugv.send_ctrl(steering, ugv.MAX_SPEED)
            cv2.arrowedLine(
                frame,
                (width // 2, height - 30),
                target,
                (0, 0, 255),
                2,
                tipLength=0.3,
            )
            print(f"[FSM: {fsm_state}] Steering: {steering:.3f}")
        else:
            print(f"[FSM: {fsm_state}] No target point found → SEARCH")
            fsm_state = STATE_SEARCH
            fsm_timer_start = time.time()

        out.write(frame)
        time.sleep(0.05)

    ugv.send_ctrl(0.0, 0.0)
    send_to_simulator(fsm_state, 0.0, 0.0)
    camera.cap.release()
    out.release()
    print("[END] Patrol complete.")


def main():
    manager = Manager()
    shared_command = manager.dict()
    shared_command["value"] = None

    patrol_process = Process(target=patrol_loop, args=(shared_command,))
    command_process = Process(target=command_server, args=(shared_command,))

    patrol_process.start()
    command_process.start()

    patrol_process.join()
    command_process.terminate()


if __name__ == "__main__":
    main()
    sys.stdout.close()
