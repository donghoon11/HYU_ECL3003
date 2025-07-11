import numpy as np
from dataclasses import dataclass
from enum import Enum


#
# FSM 상태 정의
#
class State(Enum):
    StandBy = 0
    MoveToPick = 1
    FindTarget = 2
    ApproachToTarget = 3
    PickTarget = 4
    MoveToPlace = 5
    PlaceTarget = 6
    MoveToBase = 7


#
# 설정값 정의 클래스
#
@dataclass(frozen=True)
class Config:
    map_size: tuple = (100, 100)  # 맵 셀 수 (행, 열)
    map_cell: float = 0.1  # 셀 하나의 실제 거리 (단위: m)
    place = {  # 주요 장소와 그리드 좌표 매핑
        "/bedroom1": (40, 15),
        "/bedroom2": (75, 10),
        "/toilet": (85, 30),
        "/enterance": (80, 50),
        "/dining": (91, 80),
        "/livingroom": (30, 85),
        "/balcony_init": (5, 65),
        "/balcony_end": (5, 20),
    }


#
# 미션 데이터 구조 (Pick & Place + 타겟명)
#
@dataclass(frozen=True)
class Mission:
    pick_location: str
    place_location: str
    target: str


#
# 시스템 상태 저장 및 FSM 제어용 컨텍스트 클래스
#
@dataclass
class Context:
    map: np.array = None
    map_loc: np.array = None
    mission: Mission = None
    state: State = State.StandBy
    state_count: int = 0

    base: tuple = None
    curr: tuple = None
    path: list = None
    path_idx: int = None

    def set_state(self, state):
        self.state = state
        self.state_count = 0

    def inc_state_counte(self):
        self.state_count += 1


#
# 센서 및 입력 데이터 저장용 클래스
#
@dataclass
class ReadData:
    localization: np.array = None
    joints: np.array = None
    img_flag: bool = False
    img: np.array = None
    cam_localization: np.array = None


#
# 로봇 제어 명령 저장 클래스
#
@dataclass
class ControlData:
    wheels_position: tuple = (
        np.deg2rad(0),
        np.deg2rad(0),
        np.deg2rad(0),
        np.deg2rad(0),
    )
    wheels_speed: tuple = (0.0, 0.0, 0.0, 0.0)
