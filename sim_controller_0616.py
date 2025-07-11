import time
import numpy as np
import socket
import json
import pickle
import threading
from util import ReadData, ControlData
from coppeliasim import Coppeliasim
import pandas as pd
import matplotlib.pyplot as plt


def plot_log(log_filename):
    try:
        df = pd.read_csv(log_filename)
        dx = df["x"].diff().fillna(0)
        dy = df["y"].diff().fillna(0)
        df["delta_dist"] = np.sqrt(dx**2 + dy**2)
        df["total_dist"] = df["delta_dist"].cumsum()

        # Trajectory plot
        plt.figure(figsize=(8, 6))
        plt.plot(df["x"], df["y"], label="Trajectory", linewidth=2)
        plt.scatter(
            df["x"].iloc[0], df["y"].iloc[0], color="green", label="Start", s=50
        )
        plt.scatter(df["x"].iloc[-1], df["y"].iloc[-1], color="red", label="End", s=50)
        plt.title("Simulated Rover Trajectory (Top-Down)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Distance plot
        plt.figure(figsize=(8, 4))
        plt.plot(df["time"], df["total_dist"], color="blue")
        plt.title("Cumulative Distance Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"[ERROR] Failed to visualize log: {e}")


# === Jetson 통신 설정 ===
JETSON_IP = ""
PORT_COMMAND = 9999
PORT_SIM_SYNC = 5007

# === 전역 변수 ===
run_flag = True
jetson_L = 0.0
jetson_R = 0.0

# === CoppeliaSim 객체 생성 ===
control_data = ControlData()
read_data = ReadData()
sim = Coppeliasim()


# === Jetson 명령 전송 ===
def send_command_to_rover(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((JETSON_IP, PORT_COMMAND))
            s.sendall(command.encode())
            print(f"[SENT] {command}")
    except Exception as e:
        print(f"[ERROR] Failed to send command: {e}")


# === Jetson → FSM 상태 및 속도 수신 ===
def run_sim_sync_server():
    global jetson_L, jetson_R
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", PORT_SIM_SYNC))
    server.listen(5)
    print(f"[SIM SERVER] Listening on port {PORT_SIM_SYNC}...")

    while run_flag:
        conn, addr = server.accept()
        with conn:
            data = conn.recv(1024).decode()
            if data:
                try:
                    fsm, l, r = data.strip().split(",")
                    jetson_L = float(l)
                    jetson_R = float(r)
                except Exception as e:
                    print(f"[ERROR] Parsing failed: {e}")


# === 사용자 명령 입력 처리 ===
def command_input_loop():
    while True:
        try:
            cmd = input("[INPUT] Command : ").strip().lower()
            if cmd in [
                "move to red zone",
                "move to blue zone",
                "move to yellow zone",
                "yellow zone start",
                "red zone start",
                "blue zone start",
                "total loop",
                "stop",
            ]:
                send_command_to_rover(cmd)
            else:
                print("[INVALID] Command not recognized.")
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT] Command input terminated.")
            break
        except Exception as e:
            print(f"[ERROR] command_input_loop crashed: {e}")
            break


# === 시뮬 제어 루프 (속도 → 포지션 변환)
def sim_control_loop():
    global jetson_L, jetson_R

    DT = 0.005
    SCALE = 0.02

    while run_flag:
        if read_data.joints is None:
            time.sleep(DT)
            continue

        control_data.wheels_speed = (jetson_L, jetson_R, jetson_L, jetson_R)

        curr_pos = read_data.joints
        delta = [s * SCALE for s in control_data.wheels_speed]
        control_data.wheels_position = tuple(c + d for c, d in zip(curr_pos, delta))

        time.sleep(DT)


# === CoppeliaSim 주행 루프
def drive_loop(log_filename="sim_run.log"):
    sim.sim.setStepping(True)
    sim.sim.startSimulation()

    total_dist = 0.0
    last_pos = None
    start_time = time.time()

    log_file = open(log_filename, "w", buffering=1)
    log_file.write("time,L,R,delta_L,delta_R,pos_L,pos_R,x,y\n")

    DT = 0.005
    SCALE = 0.02

    while sim.run_flag and run_flag:
        sim.read_youbot()
        read_data.localization = sim.read_data.localization
        read_data.joints = sim.read_data.joints

        x, y = read_data.localization[:2]
        if last_pos:
            dx = x - last_pos[0]
            dy = y - last_pos[1]
            total_dist += np.sqrt(dx**2 + dy**2)
        last_pos = (x, y)

        timestamp = time.time() - start_time
        L, R = control_data.wheels_speed[0], control_data.wheels_speed[1]
        delta_L = L * SCALE
        delta_R = R * SCALE
        pos_L = read_data.joints[0]
        pos_R = read_data.joints[1]
        log_file.write(
            f"{timestamp:.3f},{L:.3f},{R:.3f},{delta_L:.4f},{delta_R:.4f},{pos_L:.4f},{pos_R:.4f},{x:.4f},{y:.4f}\n"
        )

        sim.control_data = control_data
        sim.read_data = read_data
        sim.control_youbot()
        sim.sim.step()
        time.sleep(DT)

    sim.sim.stopSimulation()
    log_file.close()
    print(f"[SIM] Total simulated distance: {total_dist:.2f} m")


# === 메인 ===
if __name__ == "__main__":
    print("[SIM] Simulation + Jetson sync + Command interface start")

    try:
        log_filename = input("로그 파일 이름을 입력하세요 (.log 생략 가능): ").strip()
        if not log_filename:
            log_filename = "sim_run.log"
        elif not log_filename.endswith(".log"):
            log_filename += ".log"

        threading.Thread(target=run_sim_sync_server, daemon=True).start()
        threading.Thread(target=sim_control_loop, daemon=True).start()
        threading.Thread(target=command_input_loop, daemon=True).start()
        drive_loop(log_filename)  # 사용자 지정 로그 파일 사용

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl+C received. Exiting gracefully...")

    finally:
        run_flag = False
        sim.run_flag = False
        try:
            sim.sim.stopSimulation()
        except:
            pass
        print("[SIM] Shutdown complete.")
        # 시각화 자동 실행
        print(f"[VISUALIZATION] Plotting {log_filename}...")
        plot_log(log_filename)
