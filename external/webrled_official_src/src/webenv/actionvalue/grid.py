import math


class GridActionValue:
    def __init__(self):
        self.js_window_size = (1200, 1200)  # (width, height)
        self.point_num = (20, 20)  # (width, height)
        self.point_size = 400  # point_num[0] * point_num[1]
        self.point_distance = (60, 60)  # js_window_size / point_num
        self.point_start_position = (30, 30)  # point_distance / 2
        self.point_end_position = (1170, 1170)  # js_window_size - point_distance / 2
        self.point_round_radius = 90  # point_distance * 1.5
        self.point_round_radius_squre = 8100

    # Calculate
    @staticmethod
    def _point_distance_squre(p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def _one_dimensional_index_to_coordinate(self, n: int, axis=0):
        # axis: 0 is x, 1 is y
        return n * self.point_distance[axis] + self.point_start_position[axis]

    def _one_dimensional_coordinate_to_index(self, n, axis=0):
        return int((n - self.point_start_position[axis]) // self.point_distance[axis])

    def _is_border_on_line(self, border, axis=0):
        return ((border - self.point_start_position[axis]) % self.point_distance[axis]) == 0

    def _border_limit(self, border, axis=0):
        return min(self.point_end_position[axis], max(self.point_start_position[axis], border))

    def _point_index_to_coordinate(self, point_index: int):

        x = point_index % self.point_num[0]
        y = point_index // self.point_num[1]

        x = self._one_dimensional_index_to_coordinate(x, 0)
        y = self._one_dimensional_index_to_coordinate(y, 1)
        return x, y

    def _point_coordinate_to_index(self, point_coordinate):
        x, y = point_coordinate
        x = self._one_dimensional_coordinate_to_index(x, 0)
        y = self._one_dimensional_coordinate_to_index(y, 1)
        return y * self.point_num[0] + x

    def get_action_value_from_point(self, action_info, point_value):
        elem_value = 0

        elem_center_x = (action_info["left"] + action_info["right"]) / 2
        elem_center_y = (action_info["top"] + action_info["bottom"]) / 2

        # show
        # self.draw_round(elem_center_x, elem_center_y)

        r_left, r_right, r_top, r_bottom = 0, 0, 0, 0

        r_left_index = self._one_dimensional_coordinate_to_index((elem_center_x - self.point_round_radius), 0)
        if not self._is_border_on_line(elem_center_x - self.point_round_radius, 0):
            r_left_index += 1
        r_left = self._one_dimensional_index_to_coordinate(r_left_index, 0)
        r_left = self._border_limit(r_left, 0)

        r_right_index = self._one_dimensional_coordinate_to_index(elem_center_x + self.point_round_radius, 0)
        r_right = self._one_dimensional_index_to_coordinate(r_right_index, 0)
        r_right = self._border_limit(r_right, 0)

        r_top_index = self._one_dimensional_coordinate_to_index(elem_center_y - self.point_round_radius, 1)
        if not self._is_border_on_line(elem_center_y - self.point_round_radius, 1):
            r_top_index += 1
        r_top = self._one_dimensional_index_to_coordinate(r_top_index, 1)
        r_top = self._border_limit(r_top, 1)

        r_bottom_index = self._one_dimensional_coordinate_to_index(elem_center_y + self.point_round_radius, 1)
        r_bottom = self._one_dimensional_index_to_coordinate(r_bottom_index, 1)
        r_bottom = self._border_limit(r_bottom, 1)

        for point_x in range(r_left, r_right + 1, self.point_distance[0]):
            for point_y in range(r_top, r_bottom + 1, self.point_distance[1]):
                distance_squre = self._point_distance_squre((elem_center_x, elem_center_y), (point_x, point_y))
                if distance_squre < self.point_round_radius_squre:
                    elem_value += point_value[0][self._point_coordinate_to_index((point_x, point_y))]
        return elem_value

    def get_covered_points_from_action(self, action_info: dict):
        covered_points = {}
        elem_center_x = (action_info["left"] + action_info["right"]) / 2
        elem_center_y = (action_info["top"] + action_info["bottom"]) / 2

        # show
        # self.draw_round(elem_center_x, elem_center_y)

        r_left, r_right, r_top, r_bottom = 0, 0, 0, 0

        r_left_index = self._one_dimensional_coordinate_to_index((elem_center_x - self.point_round_radius), 0)
        if not self._is_border_on_line(elem_center_x - self.point_round_radius, 0):
            r_left_index += 1
        r_left = self._one_dimensional_index_to_coordinate(r_left_index, 0)
        r_left = self._border_limit(r_left, 0)

        r_right_index = self._one_dimensional_coordinate_to_index(elem_center_x + self.point_round_radius, 0)
        r_right = self._one_dimensional_index_to_coordinate(r_right_index, 0)
        r_right = self._border_limit(r_right, 0)

        r_top_index = self._one_dimensional_coordinate_to_index(elem_center_y - self.point_round_radius, 1)
        if not self._is_border_on_line(elem_center_y - self.point_round_radius, 1):
            r_top_index += 1
        r_top = self._one_dimensional_index_to_coordinate(r_top_index, 1)
        r_top = self._border_limit(r_top, 1)

        r_bottom_index = self._one_dimensional_coordinate_to_index(elem_center_y + self.point_round_radius, 1)
        r_bottom = self._one_dimensional_index_to_coordinate(r_bottom_index, 1)
        r_bottom = self._border_limit(r_bottom, 1)

        nearest_point, nearest_point_ratio = None, -1

        for point_x in range(r_left, r_right + 1, self.point_distance[0]):
            for point_y in range(r_top, r_bottom + 1, self.point_distance[1]):
                distance_squre = self._point_distance_squre((elem_center_x, elem_center_y), (point_x, point_y))
                if distance_squre < self.point_round_radius_squre:  # todo to be optimized
                    point_index = self._point_coordinate_to_index((point_x, point_y))
                    point_ratio = math.sqrt(1 - distance_squre / self.point_round_radius_squre)
                    covered_points[point_index] = point_ratio
                    if point_ratio > nearest_point_ratio:
                        nearest_point, nearest_point_ratio = point_index, point_ratio
        return covered_points, nearest_point