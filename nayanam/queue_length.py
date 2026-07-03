import numpy as np
from collections import defaultdict
from shapely.geometry import Point
import time

class QueueLengthCalculator:
    def __init__(self, fps=8, polygon=None, stationary_thresh=5):
        
        self.vehicle_history = defaultdict(list)
        self.reference_vehicle_id = None
        self.fps = fps
        self.polygon = polygon
        self.stationary_frames = int(2 * fps)
        self.stationary_thresh = stationary_thresh
        self.stationary_vehicles = {}
        self.vehicle_queue = 0
        self.start1 = (0,0)
        self.end1 = (0,0)

        self.current_second_lengths = []

        self.last_average_queue_length = 0

        self.last_second_timestamp = int(time.monotonic())

    def update(self, tracked_objects, vout_ps, real_length):
        
        now = int(time.monotonic())

        # Clear if movement detected (vout_ps > 0)
        if vout_ps > 0:
            self.stationary_vehicles.clear()

        for obj in tracked_objects:
            obj_id = obj['id']
            cx, cy, cz, ca, cb = obj['centroid']
            centroid = (cx, cy, cz)

            # Add new entry to history
            self.vehicle_history[obj_id].append({'centroid': centroid, 'time': now})

            # Keep only last 7 sec entries
            self.vehicle_history[obj_id] = [
                e for e in self.vehicle_history[obj_id] if now - e['time'] < 7
            ]

            # Check stationary & inside polygon
            if self.is_stationary(obj_id) and self.is_inside_polygon(centroid):
                self.stationary_vehicles[obj_id] = centroid

        # Cleanup old vehicles
        for vid in list(self.vehicle_history.keys()):
            if now - self.vehicle_history[vid][-1]['time'] > 7:
                del self.vehicle_history[vid]
                if vid == self.reference_vehicle_id:
                    self.reference_vehicle_id = None

        # Reset reference vehicle if moved
        if self.reference_vehicle_id and not self.is_stationary(self.reference_vehicle_id):
            self.reference_vehicle_id = None

        # Assign reference vehicle
        if not self.reference_vehicle_id and self.stationary_vehicles:
            self.reference_vehicle_id = self.find_reference_vehicle()


        static_length = 0

        if self.reference_vehicle_id and self.reference_vehicle_id in self.stationary_vehicles:
            start = self.stationary_vehicles[self.reference_vehicle_id]
            end = self.find_highest_stationary_vehicle()

            if start and end:
                self.start1 = (start[0],start[1])
                self.end1 = (end[0],end[2])
                homography_pixel = real_length.calculate_homography()
                static_length = real_length.calculate_real_world_distance(homography_pixel, self.start1, self.end1)


        final_length = static_length

        self.current_second_lengths.append(final_length)

        # ---------------- RETURN AVERAGE LENGTH EVERY 1 SECOND ----------------

        if now - self.last_second_timestamp >= 1:

            if len(self.current_second_lengths) > 0:
                self.last_average_queue_length = sum(self.current_second_lengths) / len(self.current_second_lengths)
            else:
                self.last_average_queue_length = 0

            # Reset for new second
            self.current_second_lengths.clear()
            self.last_second_timestamp = now
        self.vehicle_queue = len(self.stationary_vehicles)

        return self.last_average_queue_length


    def is_stationary(self, obj_id):
        history = self.vehicle_history.get(obj_id, [])
        if len(history) < self.stationary_frames:
            return False

        first = np.array(history[-self.stationary_frames]['centroid'])
        last = np.array(history[-1]['centroid'])

        return abs(first[1] - last[1]) < self.stationary_thresh

    def is_inside_polygon(self, centroid):
        if self.polygon is None:
            return True
        return self.polygon.contains(Point(centroid))

    def find_reference_vehicle(self):
        closest_id = None
        max_y = -1
        for obj_id, centroid in self.stationary_vehicles.items():
            if centroid[1] > max_y:
                closest_id = obj_id
                max_y = centroid[1]
        return closest_id

    def find_highest_stationary_vehicle(self):
        end_coord = None
        min_y = float('inf')
        for centroid in self.stationary_vehicles.values():
            if centroid[2] < min_y:
                min_y = centroid[2]
                end_coord = centroid
        return end_coord

    # ---------------- DYNAMIC QUEUE ---------------------
    def get_dynamic_queue_length(self, tracked_objects):
        if len(tracked_objects) < 2:
            return 0

        sorted_objs = sorted(tracked_objects, key=lambda o: o['centroid'][1])

        nearest = sorted_objs[0]
        farthest = sorted_objs[-1]

        cx1, cy1, cz1, ca1, cb1 = nearest['centroid']
        cx2, cy2, cz2, ca2, cb2 = farthest['centroid']

        diff = (cy2 - cz1)
        dist_meters = (diff / 100) * 3.5 

        return dist_meters

    def get_vehicle_queue(self):
        return self.vehicle_queue

    def get_coordinates_queue(self):
        return self.start1, self.end1
