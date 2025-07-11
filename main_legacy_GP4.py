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
import traceback

# 로그 저장 설정
log_path = input("로그 파일 이름 (.log 포함): ").strip()
video_path = input("비디오 파일 이름 (.avi 포함): ").strip()

sys.stdout = open(log_path, "w", buffering=1)
sys.stderr = sys.stdout

# === 모델 로드 ===
road_model = YOLO("/home/ircv14/HYU-ECL3003/rover/models/goat_ver2_sNr.engine")
# road_model = YOLO("/home/ircv14/HYU-ECL3003/rover/models/goat_ver2_r.engine")
# road_model = YOLO("/home/ircv14/HYU-ECL3003/rover/models/goat_ver1_s.engine")

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
STATE_Y_P_B = "Y_P_B"
STATE_B_G_Y = "B_G_Y"
STATE_Y_P_R = "Y_P_R"
STATE_R_G_Y = "R_G_Y"
STATE_SEARCH = "SEARCH"
STATE_STOP = "STOP"

# === 설정 ===
SEARCH_DURATION = 0.5
FOCAL_LENGTH_PX = 1650
CONFIDENCE_THRESHOLD = 0.5
DOWNSAMPLE = 2
FORWARD_DURATION = 2.2  # 화이트 라인 전진 시간

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
        sim_sock.connect((YOUR_IP, 5007))	# 로컬PC_Ip, 포트번호
        msg = f"{fsm_state},{L:.3f},{R:.3f}"
        sim_sock.sendall(msg.encode("utf-8"))
        print(f"[SIM SYNC] Sent: {fsm_state}, L={L:.3f}, R={R:.3f}")
        sim_sock.close()
    except Exception as e:
        print(f"[SIM SYNC] Send failed: {e}")


class UGVAutoController:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200):
        self.base_ctrl = BaseController(port, baudrate)
        self.MAX_STEER = 0.88  # mean : 0.95
        self.MAX_SPEED = 0.4  # 0.5로 초기화. 이후 send_ctrl에 의해 수정
        self.STEER_GAIN = 0.005

    def send_ctrl(self, steering, speed):
        steer_val = clip(steering, self.MAX_STEER)

        # 속도 계산은 raw steering 값 사용 (클리핑 전)
        abs_raw_steer = abs(steering)
        # 속도 계산을 위한 스티어링 정규화 (실제 사용 범위에 맞춤)
        normalized_steer = min(
            abs_raw_steer / 5.2, self.MAX_STEER
        )  # 5.0 이상이면 최대로 간주

        # 동적 속도 계산 (직진 시 느리게, 코너링 시 빠르게)
        dynamic_speed = 0.25 + normalized_steer * 0.25  # 0.25~0.5 범위
        dynamic_speed = min(dynamic_speed, 0.5)  # 최대 0.5로 제한

        # 입력 speed와 동적 속도 중 작은 값 사용 (안전장치)
        final_speed = min(speed, dynamic_speed)

        left_ratio = 1.0 - steer_val
        right_ratio = 1.0 + steer_val
        L = clip(final_speed * left_ratio, final_speed)
        R = clip(final_speed * right_ratio, final_speed)

        if final_speed < 0:
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


def draw_hud(frame, fsm_state, L, R, steering, speed, width):
    cv2.putText(
        frame,
        f"FSM: {fsm_state}",
        (width - 600, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 0),
        4,
    )
    cv2.putText(
        frame,
        f"L={L:.2f} R={R:.2f}",
        (width - 600, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 0),
        4,
    )
    cv2.putText(
        frame,
        f"steer={steering:.3f}",
        (width - 600, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 0),
        4,
    )
    cv2.putText(
        frame,
        f"speed={speed:.2f}",
        (width - 600, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 0),
        4,
    )


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

    # 순환 방문을 위한 이전 존 기록
    last_zone = None  # 마지막으로 있었던 존 (Y, R, B)

    # 화이트 라인 전진 타이머 추가
    white_forward_timer = {}

    camera = CSICamera(
        capture_width=3280, capture_height=2464, downsample=DOWNSAMPLE, capture_fps=21
    )

    ugv = UGVAutoController()
    width, height = 3280, 2464
    out = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"MJPG"), 21, (width, height)
    )

    # 시작 존 설정
    if fsm_state == STATE_Y:
        last_zone = "Y"
    elif fsm_state == STATE_R:
        last_zone = "R"
    elif fsm_state == STATE_B:
        last_zone = "B"

    while True:
        print(f"[FSM] Current State: {fsm_state}, Last Zone: {last_zone}")
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
                    white_forward_timer[STATE_Y_W_R] = time.time()  # 타이머 시작
                    print("[FSM] Transition: Y → Y_W_R (starting with 0.1s forward)")
                    continue
                elif cmd == "move to blue zone" and "white" in detections:
                    fsm_state = STATE_Y_W_B
                    white_forward_timer[STATE_Y_W_B] = time.time()  # 타이머 시작
                    print("[FSM] Transition: Y → Y_W_B (starting with 0.1s forward)")
                    continue
            target_points = extract_center_points(results, road_model, "yellow")
            if target_points:
                last_valid_state = fsm_state

        elif fsm_state == STATE_Y_W_R:
            # 타이머 확인: 0.1초 미만이면 직진
            if (
                STATE_Y_W_R in white_forward_timer
                and time.time() - white_forward_timer[STATE_Y_W_R] < FORWARD_DURATION
            ):
                # 0.1초 동안 무지성 직진
                L, R = ugv.send_ctrl(0.0, 0.4)  # 스티어링=0, 직진
                send_to_simulator(fsm_state, L, R)
                draw_hud(frame, fsm_state + "_FWD", L, R, 0.0, 0.4, width)
                print(
                    f"[FSM: {fsm_state}] Forward phase: {time.time() - white_forward_timer[STATE_Y_W_R]:.2f}s"
                )
                out.write(frame)
                time.sleep(0.05)
                continue
            else:
                # 0.1초 경과 후 일반 라인 따라가기
                if "red" in detections:
                    fsm_state = STATE_R
                    last_zone = "Y"
                    last_valid_state = STATE_R
                    if STATE_Y_W_R in white_forward_timer:
                        del white_forward_timer[STATE_Y_W_R]
                    print("[FSM] Transition: Y_W_R → R")
                    continue

                target_points = extract_center_points(results, road_model, "white")
                if target_points:
                    last_valid_state = fsm_state

        elif fsm_state == STATE_Y_W_B:
            if (
                STATE_Y_W_B in white_forward_timer
                and time.time() - white_forward_timer[STATE_Y_W_B] < FORWARD_DURATION
            ):
                L, R = ugv.send_ctrl(0.0, 0.4)
                send_to_simulator(fsm_state, L, R)
                draw_hud(frame, fsm_state + "_FWD", L, R, 0.0, 0.4, width)
                print(
                    f"[FSM: {fsm_state}] Forward phase: {time.time() - white_forward_timer[STATE_Y_W_B]:.2f}s"
                )
                out.write(frame)
                time.sleep(0.05)
                continue
            else:
                if "blue" in detections:
                    fsm_state = STATE_B
                    last_zone = "Y"
                    last_valid_state = STATE_B
                    if STATE_Y_W_B in white_forward_timer:
                        del white_forward_timer[STATE_Y_W_B]
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
                white_forward_timer[STATE_B_W_Y] = time.time()
                print("[FSM] Transition: B → B_W_Y (starting with 0.1s forward)")
                continue
            target_points = extract_center_points(results, road_model, "blue")
            if target_points:
                last_valid_state = fsm_state

        elif fsm_state == STATE_B_W_Y:
            if (
                STATE_B_W_Y in white_forward_timer
                and time.time() - white_forward_timer[STATE_B_W_Y] < FORWARD_DURATION
            ):
                L, R = ugv.send_ctrl(0.0, 0.4)
                send_to_simulator(fsm_state, L, R)
                draw_hud(frame, fsm_state + "_FWD", L, R, 0.0, 0.4, width)
                print(
                    f"[FSM: {fsm_state}] Forward phase: {time.time() - white_forward_timer[STATE_B_W_Y]:.2f}s"
                )
                out.write(frame)
                time.sleep(0.05)
                continue
            else:
                if "yellow" in detections:
                    fsm_state = STATE_Y
                    last_zone = "B"
                    last_valid_state = STATE_Y
                    if STATE_B_W_Y in white_forward_timer:
                        del white_forward_timer[STATE_B_W_Y]
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
                white_forward_timer[STATE_R_W_Y] = time.time()
                print("[FSM] Transition: R → R_W_Y (starting with 0.1s forward)")
                continue
            target_points = extract_center_points(results, road_model, "red")
            if target_points:
                last_valid_state = fsm_state

        elif fsm_state == STATE_R_W_Y:
            if (
                STATE_R_W_Y in white_forward_timer
                and time.time() - white_forward_timer[STATE_R_W_Y] < FORWARD_DURATION
            ):
                L, R = ugv.send_ctrl(0.0, 0.4)
                send_to_simulator(fsm_state, L, R)
                draw_hud(frame, fsm_state + "_FWD", L, R, 0.0, 0.4, width)
                print(
                    f"[FSM: {fsm_state}] Forward phase: {time.time() - white_forward_timer[STATE_R_W_Y]:.2f}s"
                )
                out.write(frame)
                time.sleep(0.05)
                continue
            else:
                if "yellow" in detections:
                    fsm_state = STATE_Y
                    last_zone = "R"
                    last_valid_state = STATE_Y
                    if STATE_R_W_Y in white_forward_timer:
                        del white_forward_timer[STATE_R_W_Y]
                    print("[FSM] Transition: R_W_Y → Y")
                    continue

                target_points = extract_center_points(results, road_model, "white")
                if target_points:
                    last_valid_state = fsm_state

        # Total loop FSM - 순환 방문 로직
        if total_loop_mode:
            # 초기 전환: 목표 색상에서 중간 색상 감지 시
            if fsm_state == STATE_Y and "purple" in detections:
                # 순환 방문: Y에서 왔을 때는 R 또는 B로 (마지막 존에 따라)
                if last_zone == "R":  # R→Y→B 순환
                    fsm_state = STATE_Y_P_B
                    print(
                        f"[FSM] Transition: Y → Y_P_B (purple detected, cyclic R→Y→B)"
                    )
                elif last_zone == "B":  # B→Y→R 순환
                    fsm_state = STATE_Y_P_R
                    print(
                        f"[FSM] Transition: Y → Y_P_R (purple detected, cyclic B→Y→R)"
                    )
                else:  # 초기 상태이거나 불분명한 경우 기본값
                    fsm_state = STATE_Y_P_R
                    print(f"[FSM] Transition: Y → Y_P_R (purple detected, default)")

            elif fsm_state == STATE_R and "green" in detections:
                fsm_state = STATE_R_G_Y
                print("[FSM] Transition: R → R_G_Y (green detected)")

            elif fsm_state == STATE_B and "green" in detections:
                fsm_state = STATE_B_G_Y
                print("[FSM] Transition: B → B_G_Y (green detected)")

            # 중간 색상이 더 이상 감지되지 않을 때 다음 상태로 전환
            elif fsm_state == STATE_Y_P_R and "purple" not in detections:
                fsm_state = STATE_R
                last_zone = "Y"
                print("[FSM] Transition: Y_P_R → R (purple lost)")
            elif fsm_state == STATE_R_G_Y and "green" not in detections:
                fsm_state = STATE_Y
                last_zone = "R"
                print("[FSM] Transition: R_G_Y → Y (green lost)")
            elif fsm_state == STATE_Y_P_B and "purple" not in detections:
                fsm_state = STATE_B
                last_zone = "Y"
                last_valid_state = STATE_B  # 상태 전환 시 즉시 업데이트
                print("[FSM] Transition: Y_P_B → B (purple lost)")
            elif fsm_state == STATE_B_G_Y and "green" not in detections:
                fsm_state = STATE_Y
                last_zone = "B"
                last_valid_state = STATE_Y  # 상태 전환 시 즉시 업데이트
                print("[FSM] Transition: B_G_Y → Y (green lost)")

        # 각 상태에서의 타겟 포인트 설정
        if fsm_state == STATE_Y_P_R:
            target_points = extract_center_points(results, road_model, "purple")
        elif fsm_state == STATE_R_G_Y:
            target_points = extract_center_points(results, road_model, "green")
        elif fsm_state == STATE_Y_P_B:
            target_points = extract_center_points(results, road_model, "purple")
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

        # bottom 조향 처리
        if target_points:
            target_points = sorted(target_points, key=lambda pt: pt[1])
            target = max(target_points, key=lambda pt: pt[1])
            error = target[0] - (width // 2)
            steering = -error * ugv.STEER_GAIN
            L, R = ugv.send_ctrl(steering, 0.5)  # 최대 동적 속도와 동일한 값 전달
            send_to_simulator(fsm_state, L, R)
            cv2.arrowedLine(
                frame,
                (width // 2, height - 30),
                target,
                (0, 0, 255),
                2,
                tipLength=0.3,
            )
            # HUD에는 실제 계산된 속도 표시 (send_ctrl과 동일한 계산)
            abs_raw_steer = abs(steering)
            normalized_steer = min(abs_raw_steer / 20.0, 1.0)
            actual_speed = min(0.5, 0.25 + normalized_steer * 0.25)
            draw_hud(frame, fsm_state, L, R, steering, actual_speed, width)
            print(
                f"[FSM: {fsm_state}] Steering: {steering:.3f} Speed: {actual_speed:.3f}"
            )
        else:
            print(f"[FSM: {fsm_state}] No target point found → SEARCH")
            fsm_state = STATE_SEARCH
            draw_hud(frame, "SEARCH", -0.2, -0.2, 0.0, 0.2, width)
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
    try:
        main()
    except Exception as e:
        print("[MAIN ERROR]")
        traceback.print_exc()
    sys.stdout.close()
