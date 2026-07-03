from collections import defaultdict
from shapely.geometry import Point, Polygon
import time

class VehicleOutCrossCounter:
    def __init__(self, line_p1, line_p2, zone_points, fps):
        self.p1 = line_p1
        self.p2 = line_p2
        self.zone = Polygon(zone_points)
        self.fps = fps

        self.prev_side = {}
        self.per_second = defaultdict(int)

        self.start_time = int(time.monotonic())
        self.last_second_sent = -1   # track the last second for which we sent data

        # store last-frame data of zone per second
        self.zone_last_frame = {}   # sec -> dict
        self.zone_category_current = defaultdict(int)


    def _side(self, cx, cy):
        return (cx - self.p1[0]) * (self.p2[1] - self.p1[1]) - \
               (cy - self.p1[1]) * (self.p2[0] - self.p1[0])

    def update(self, objects, frame_idx, custom_names):
        """
        Call this every frame.
        """
        t_sec = int(time.monotonic()) - self.start_time

        # fresh counts for this frame
        self.zone_category_current.clear()

        for obj in objects:
            tid = obj['id']
            cx, cy, cz, ca, cb = obj['centroid']
            class_id = obj['class_id']

            # if inside zone → count
            if (
                self.zone.contains(Point(ca, cb)) or
                self.zone.contains(Point(cx, cy)) or
                self.zone.contains(Point(cx, cz))
            ):
                self.zone_category_current[custom_names[class_id]] += 1

            # crossing detection
            side_now = 1 if self._side(cx, cy) > 0 else -1
            side_prev = self.prev_side.get(tid, side_now)

            if side_now != side_prev:
                self.per_second[t_sec] += 1

            self.prev_side[tid] = side_now

        # Save this frame as "last frame of current second"
        self.zone_last_frame[t_sec] = dict(self.zone_category_current)

        return self.zone_category_current


    def get_zone_last_second(self):
        current_sec = int(time.monotonic()) - self.start_time

        # wait until we pass 1 second
        if self.last_second_sent == -1:
            self.last_second_sent = current_sec
            return None

        # If a NEW second has started → send previous second's zone data
        if current_sec > self.last_second_sent:
            previous_sec = self.last_second_sent
            self.last_second_sent = current_sec

            return self.zone_last_frame.get(previous_sec, {})

        return None


    def get_crossing_last_second(self):
        prev_sec = self.last_second_sent - 1
        if prev_sec < 0:
            return None

        return self.per_second.get(prev_sec, 0)


    def normalise_vehicle(self):
        total_normalise_vehicles = 0
        for cls, count in self.zone_category_current.items():
            if cls=='2w':
                total_normalise_vehicles+=(count*0.5)
            if cls=='3w':
                total_normalise_vehicles+=(count*1)
            if cls=='lmv':
                total_normalise_vehicles+=(count*1)
            if cls=='hmv':
                total_normalise_vehicles+=(count*3.5)
        return total_normalise_vehicles
