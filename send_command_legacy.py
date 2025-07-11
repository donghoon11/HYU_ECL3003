import socket
import sys

JETSON_IP = ""
PORT = 9999


def send_command_to_rover(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((JETSON_IP, PORT))
            s.sendall(command.encode())
            print(f"[SENT] {command}")
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    print("[INFO] Type your command. Use 'stop' or Ctrl+C to exit safely.")
    try:
        while True:
            cmd = (
                input("[INPUT] Command (move to red/blue/yellow zone, stop): ")
                .strip()
                .lower()
            )
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
                if cmd == "stop":
                    print("[INFO] Stop command sent. Exiting.")
                    break
            else:
                print("[INVALID] Command not recognized.")
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C detected. Sending 'stop' to rover...")
        send_command_to_rover("stop")
    finally:
        print("[CLEANUP] Program terminated safely.")
        sys.exit(0)
