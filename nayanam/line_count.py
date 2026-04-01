class LineVehicleCounter:
    def __init__(self, junction_name, line_coords):
        """
        junction_name: unique name for this junction (e.g., 'junction_1')
        line_coords: string "x1,y1,x2,y2" in frame coordinates
        ct: initial count (default 0)
        """
        self.junction_name = junction_name
        self.line_coords = line_coords
        # tracker_id -> {"side": last_side, "counted": bool}
        self.prev_positions = {}

    def update(self, tracker_id, center,y):
        """
        Call this for each vehicle position update in each frame.
        tracker_id: unique ID from tracker
        center: (x, y) of vehicle center
        """
        if len(self.prev_positions) >= 2000:  # keep memory bounded
            self.prev_positions.clear()

        sign = self._point_side(center)

        # First time seeing this vehicle
        if tracker_id not in self.prev_positions:
            x1,y1 = center
            if y>y1:
               self.prev_positions[tracker_id] = {"side": sign, "counted": False}
            else:
               self.prev_positions[tracker_id] = {"side": sign, "counted": True}
            return 0
        
        if sign == 0:
            return 0

        state = self.prev_positions[tracker_id]
        prev_sign = state["side"]
        counted = state["counted"]

        if prev_sign != sign:
            if not counted:  # only count once
                state["counted"] = True  # mark as already counted
            else:
                return 0
            state["side"] = sign
            return 1

        return 0

    def _point_side(self, point):
        x1, y1, x2, y2 = map(int, map(float, self.line_coords.split(',')))
        px, py = point
        val = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
        return 1 if val > 0 else (-1 if val < 0 else 0)

