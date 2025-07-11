# Copyright 2024 @With-Robot 3.5
#
# Licensed under the MIT License;
#     https://opensource.org/license/mit

from threading import Thread
import numpy as np

from util import Config, Context, ReadData, ControlData


class CarClass:
    def __init__(self, shape):
        self.config = Config()
        self.map_loc = self._build_map_loc(shape)

    def _build_map_loc(self, shape):
        map_loc = np.zeros((shape[0], shape[1], 2))
        full = shape[0] * 0.1
        map_loc[:, :, 0] = np.linspace(
            -full / 2 + 0.05, full / 2 - 0.05, shape[0]
        ).reshape(1, -1)
        full = shape[1] * 0.1
        map_loc[:, :, 1] = np.linspace(
            -full / 2 + 0.05, full / 2 - 0.05, shape[1]
        ).reshape(-1, 1)
        return map_loc

    def _calc_map_mask(self, map):
        n_row, n_col = map.shape
        walls = np.argwhere(map == 1)
        masks = np.zeros_like(map)
        masks[:3, :] = 1
        masks[-3:, :] = 1
        masks[:, :3] = 1
        masks[:, -3:] = 1
        for x, y in walls:
            masks[
                max(x - 3, 0) : min(x + 4, n_row + 1),
                max(y - 3, 0) : min(y + 4, n_col + 1),
            ] = 1
        masks *= map != 1
        map[masks > 0] = 1.0
        return map

    def _calc_map_value(self, map_mask, end):
        n_row, n_col = map_mask.shape
        v_prev = np.zeros(map_mask.shape)
        v_next = np.zeros(map_mask.shape)

        def cal_value(row, col):
            if map_mask[row, col] == 1.0:
                return -n_row * n_col
            prev_row = max(0, row - 1)
            next_row = min(n_row - 1, row + 1)
            prev_col = max(0, col - 1)
            next_col = min(n_col - 1, col + 1)

            loc_list = [
                (prev_row, col),
                (next_row, col),
                (row, prev_col),
                (row, next_col),
                (prev_row, prev_col),
                (prev_row, next_col),
                (next_row, prev_col),
                (next_row, next_col),
            ]

            loc_value = []
            for loc in loc_list:
                value = -1 + (v_prev[row, col] if map_mask[loc] == 1.0 else v_prev[loc])
                loc_value.append(value)
            return max(loc_value)

        for _ in range(1000):
            v_next.fill(0.0)
            for row in range(n_row):
                for col in range(n_col):
                    if (row, col) != end:
                        v_next[row, col] = cal_value(row, col)
            if np.sum(np.abs(v_prev - v_next)) < 0.1:
                print("planning ok ...")
                break
            v_prev, v_next = v_next, v_prev

        return v_next

    def _calc_map_path(self, context, start, end):
        map_mask = self._calc_map_mask(context.map.copy())
        map_value = self._calc_map_value(map_mask, end)

        checked = set()
        n_row, n_col = map_value.shape
        position = start
        map_path = []
        for _ in range(1000):
            map_path.append(tuple(position))
            if position == end:
                break
            row, col = position
            prev_row = max(0, row - 1)
            next_row = min(n_row - 1, row + 1)
            prev_col = max(0, col - 1)
            next_col = min(n_col - 1, col + 1)

            items = [
                (prev_row, col),
                (next_row, col),
                (row, prev_col),
                (row, next_col),
                (prev_row, prev_col),
                (prev_row, next_col),
                (next_row, prev_col),
                (next_row, next_col),
            ]
            loc_list = [loc for loc in items if loc not in checked]
            for loc in loc_list:
                checked.add(loc)

            if not loc_list:
                break

            loc_value = np.array([map_value[loc] for loc in loc_list])
            index = np.argmax(loc_value)
            position = loc_list[index]

        return map_path

    def _calc_map_path_cb(self, context, start, end):
        context.path = self._calc_map_path(context, start, end)
        context.path_idx = 0

    def _calc_angle_diff(self, target, curr):
        angle = target - curr
        if angle < -np.pi:
            angle += 2 * np.pi
        if angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def _calc_farthest_path(self, context: Context, read_data: ReadData, curr: tuple):
        path_idx = context.path_idx
        for i in range(context.path_idx, len(context.path)):
            target = context.path[i]
            if curr[0] == target[0] or curr[1] == target[1]:
                path_idx = i
            elif abs(curr[0] - target[0]) == abs(curr[1] - target[1]):
                path_idx = i
            else:
                break
        return path_idx

    def _follow_path(
        self, context: Context, read_data: ReadData, control_data: ControlData
    ):
        if len(context.path) <= context.path_idx:
            context.path = None
            context.path_idx = None
            control_data.wheels_position = tuple(read_data.joints[:4])
            return True

        curr = self.point_to_gird(read_data.localization[:2])
        if curr != context.curr:
            context.path_idx = self._calc_farthest_path(context, read_data, curr)

        target = context.path[context.path_idx]
        if target == curr:
            context.path_idx += 1
        else:
            diff = self.map_loc[target] - read_data.localization[:2]
            curr_z = read_data.localization[5] - np.pi
            target_z = np.arctan2(diff[1], diff[0])
            angle = self._calc_angle_diff(target_z, curr_z)
            distance = np.linalg.norm(diff)

            if abs(angle) > 0.1:
                angle *= 0.5
                control_data.wheels_position = (
                    read_data.joints[0] + angle,  # fl
                    read_data.joints[1] - angle,  # fr
                    read_data.joints[2] + angle,  # bl
                    read_data.joints[3] - angle,  # br
                )
            else:
                distance *= np.pi
                control_data.wheels_position = (
                    read_data.joints[0] + distance,  # fl
                    read_data.joints[1] + distance,  # fr
                    read_data.joints[2] + distance,  # bl
                    read_data.joints[3] + distance,  # br
                )

    def point_to_gird(self, point):
        n_row, n_col = self.map_loc.shape[:2]
        norm = np.linalg.norm(self.map_loc - point, axis=-1)
        index = np.argmin(norm.reshape(-1))
        row, col = index // n_row, index % n_col
        return row, col

    def move_to_pick(
        self, context: Context, read_data: ReadData, control_data: ControlData
    ):
        if context.state_count == 1:
            read_data.scan_flg = True
            read_data.img_flag = False

            start = self.point_to_gird(read_data.localization[:2])
            end = self.config.place[context.mission.pick_location]
            t = Thread(target=self._calc_map_path_cb, args=(context, start, end))
            t.start()
        elif context.path:
            return self._follow_path(context, read_data, control_data)

    def move_to_place(
        self, context: Context, read_data: ReadData, control_data: ControlData
    ):
        if context.state_count == 1:
            read_data.scan_flg = True
            read_data.img_flag = False

            start = self.point_to_gird(read_data.localization[:2])
            end = self.config.place[context.mission.place_location]
            t = Thread(target=self._calc_map_path_cb, args=(context, start, end))
            t.start()
        elif context.path:
            return self._follow_path(context, read_data, control_data)

    def move_to_base(
        self, context: Context, read_data: ReadData, control_data: ControlData
    ):
        if context.state_count == 1:
            read_data.scan_flg = True
            read_data.img_flag = False

            start = self.point_to_gird(read_data.localization[:2])
            end = context.base
            t = Thread(target=self._calc_map_path_cb, args=(context, start, end))
            t.start()
        elif context.path:
            return self._follow_path(context, read_data, control_data)
