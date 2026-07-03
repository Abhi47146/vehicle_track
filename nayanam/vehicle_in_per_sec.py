from collections import defaultdict
import time

class VehicleInCrossCounter:
    def __init__(self, line_p1, line_p2, keep_seconds=30):
        
        self.p1 = line_p1
        self.p2 = line_p2
        self.prev_side = {}               # track_id -> last side of line
        self.per_second = defaultdict(int)  # second_offset -> count
        self.last_reported_sec = int(-1)  
        self.current_display_count = 0    
        self.start_time = int(time.monotonic())
        self.keep_seconds = keep_seconds

    def _side(self, cx, cy):
        return (cx - self.p1[0]) * (self.p2[1] - self.p1[1]) - \
               (cy - self.p1[1]) * (self.p2[0] - self.p1[0])

    def _prune_old(self, upto_sec):
        """
        Remove entries older than (upto_sec - keep_seconds)
        """
        cutoff = upto_sec - self.keep_seconds
        # gather keys to delete to avoid runtime dict change
        to_del = [k for k in self.per_second.keys() if k < cutoff]
        for k in to_del:
            del self.per_second[k]

    def update(self, objects):
        
        # integer second offset since start
        t_sec = int(time.monotonic()) - self.start_time

        # process objects for *current* second t_sec
        for obj in objects:
            tid = obj['id']
            centroid = obj.get('centroid')
            if centroid is None:
                continue

            try:
                cx, cy, cz, ca, cb = centroid
            except Exception:
                try:
                    cx, cy = centroid[0], centroid[1]
                except Exception:
                    continue

            side_now = 1 if self._side(cx, cy) > 0 else -1
            side_prev = self.prev_side.get(tid, side_now)

            if side_now != side_prev:
                self.per_second[t_sec] += 1

            self.prev_side[tid] = side_now

       
        if t_sec > self.last_reported_sec:
            prev_sec = t_sec - 1
            if prev_sec >= 0:
                self.current_display_count = self.per_second.get(prev_sec, 0)
            else:
                self.current_display_count = 0

            self.last_reported_sec = t_sec

            self._prune_old(t_sec)

        return self.current_display_count
