import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from util import ReadData, ControlData


class Coppeliasim:
    def __init__(self):
        self.sim = RemoteAPIClient().require("sim")
        self.read_data = ReadData()
        self.control_data = ControlData()
        self.run_flag = True

        # 기준 위치 객체
        self.robot_ref = self.sim.getObject("/dummy")

        # 바퀴 조인트 핸들
        self.joints = [
            self.sim.getObject("/fl_joint"),
            self.sim.getObject("/fr_joint"),
            self.sim.getObject("/bl_joint"),
            self.sim.getObject("/br_joint"),
        ]

        # 카메라 핸들
        self.camera_1 = self.sim.getObject("/camera_1")

        # 조인트 제어 모드 설정
        self.set_joint_ctrl_mode(self.joints, self.sim.jointdynctrl_position)

    def set_joint_ctrl_mode(self, objects, ctrl_mode):
        for obj in objects:
            self.sim.setObjectInt32Param(
                obj, self.sim.jointintparam_dynctrlmode, ctrl_mode
            )

    def read_youbot(self):
        # 로봇 위치 및 자세
        pos = self.sim.getObjectPosition(self.robot_ref)
        ori = self.sim.getObjectOrientation(self.robot_ref)
        self.read_data.localization = np.array(pos + ori)

        # 카메라 위치 및 이미지
        pos = self.sim.getObjectPosition(self.camera_1)
        ori = self.sim.getObjectOrientation(self.camera_1)
        self.read_data.cam_localization = np.array(pos + ori)

        if self.read_data.img_flag:
            result = self.sim.getVisionSensorImg(self.camera_1)
            img = np.frombuffer(result[0], dtype=np.uint8)
            img = img.reshape((result[1][1], result[1][0], 3))
            img = cv2.flip(img, 0)
            self.read_data.img = img
        else:
            self.read_data.img = None

        # 조인트 상태 읽기
        joints = [self.sim.getJointPosition(j) for j in self.joints]
        self.read_data.joints = np.array(joints)

    def control_youbot(self):
        # 바퀴 제어
        if self.control_data.wheels_position is not None:
            for i, wheel in enumerate(self.control_data.wheels_position):
                curr = self.read_data.joints[i]
                diff = min(abs(wheel - curr), np.pi / 8)
                target = curr + diff if curr < wheel else curr - diff
                self.sim.setJointTargetPosition(self.joints[i], target)

    def run(self, callback):
        self.sim.setStepping(True)
        self.sim.startSimulation()

        while self.run_flag:
            self.read_youbot()
            callback(self.read_data, self.control_data)
            self.control_youbot()
            self.sim.step()

        self.sim.stopSimulation()
