from datetime import datetime
import numpy as np
from configparser import ConfigParser

class Polygon():
    """
    Creates a zone for given coordinates, 
    and has methods which gives information regarding whether a  vehicle is in the zone or not
    """

    def __init__(self,points:list) -> None:
        self.INT_MAX = 10000
        self.points = points

    def onSegment(self,p:tuple, q:tuple, r:tuple) -> bool:
        if ((q[0] <= max(p[0], r[0])) &
            (q[0] >= min(p[0], r[0])) &
            (q[1] <= max(p[1], r[1])) &
            (q[1] >= min(p[1], r[1]))):
            return True
        return False
    
    def orientation(self,p:tuple, q:tuple, r:tuple) -> int:
        val = (((q[1] - p[1]) *
                (r[0] - q[0])) -
            ((q[0] - p[0]) *
                (r[1] - q[1])))
        if val == 0:
            return 0
        if val > 0:
            return 1
        else:
            return 2 

    def doIntersect(self,p1, q1, p2, q2):
    
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)
    
        if (o1 != o2) and (o3 != o4):
            return True
        
        if (o1 == 0) and (self.onSegment(p1, p2, q1)):
            return True
    
        if (o2 == 0) and (self.onSegment(p1, q2, q1)):
            return True

        if (o3 == 0) and (self.onSegment(p2, p1, q2)):
            return True
    
        if (o4 == 0) and (self.onSegment(p2, q1, q2)):
            return True
    
        return False
 
    def contains_points(self, p) -> bool:
        n = len(self.points)
        p = p[0]

        if n < 3:
            return False

        extreme = (self.INT_MAX, p[1])
        count = i = 0
        
        while True:
            next = (i + 1) % n
            
            if (self.doIntersect(self.points[i],
                            self.points[next],
                            p, extreme)):
                                
                if self.orientation(self.points[i], p,
                            self.points[next]) == 0:
                    return self.onSegment(self.points[i], p,
                                    self.points[next])
                                    
                count += 1
                
            i = next
            
            if (i == 0):
                break
            
        return count % 2 == 1

class ZoneAssignment:
    """
    Tracks and Assigns zone to the vehicles when vehicle enters the zone
    Args:
        arm_id(str):
        config (NyanamConfig object): contains all the data from configfiles
    Attributes:
        config (NyanamConfig object): contains all the data from configfiles
        logger (Logger object): logger
        zones (dict): keys as zone ids and values as list of dimensions
        polygons (list): list of Polygon objects
        tracked_objects(list): list of all tracked objects
        zone_coords(dict): keys as zone ids and values as list of dimensions

    """
    def __init__(self,arm_id,zone_configfile):
        # self.args = get_config(config, overrides)
        # self.arm_id = self.args.arm_id
        #self.zones = config.zones[arm_id]
        self.conf = ConfigParser()
        self.config_file = zone_configfile
        self.conf.read(self.config_file)
       
        self.zone_cordinates=self.conf.get(arm_id,'zone_dimensions_1')
        self.list_of_zone_coordinates = [float(coord) for coord in self.zone_cordinates.split(',')]
        self.zones={1: self.list_of_zone_coordinates}
        self.spd={1: {'lsg_1': [0, 0, 0, 0], 'dis_1': 0}}
        self.polygons = []        
        self.tracked_objects = []
        self.zone_coords = {}
        self.sp = {}
        self.tim = {}
        c = 1
        self.total_count=0
        self.debug=True
        for _id,zone in self.zones.items():
            p = Polygon([(zone[0],zone[1]), (zone[2],zone[3]), (zone[4],zone[5]), (zone[6],zone[7])])
            self.polygons.append(p)
            self.zone_coords[_id] = np.array([[zone[0],zone[1]], [zone[2],zone[3]], [zone[4],zone[5]], [zone[6],zone[7]]],np.int32)
            tlsg = self.spd[c]['lsg_' + str(c)]
            tdis = self.spd[c]['dis_' + str(c)]
            self.sp[c] = {"lsg":[[tlsg[0],tlsg[1]], [tlsg[2],tlsg[3]], [zone[6],zone[7]], [zone[0],zone[1]]],
                    "dis":tdis
            }
            c += 1
        
        

    def check_vehicle_in_zone(self,x,y):
        """
        Checks if vehicle is any of the zone.

        Args:
            id (int): 
            x (float): center x of the bbox
            y (float): center y of the bbox

        Returns:
            zone_id: return zone number in which the vehicle is present
        """
        for n,p in enumerate(self.polygons):
            pos = p.contains_points([(x, y)])
            if pos:
                return 1
        return 0

    def generate_vbv(self,object,class_name,event,speed):
        """
        Generates vbv if the object is detected

        Args:
            object (int): object id
            class_name (str): object class
            event (str): vehicle entry or vehicle exit
            lane (int): zone number
            queue (Queue): puts the data into socket queue
            speed (m/s): speed of the vehicle
        """
        vbv = {
                "TimeStamp":str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "SCN":self.arm_id,
                "ObjectID":str(object),
                "VehicleClass":class_name,
                "Event":event,
                "Speed":str(speed)
            }
        #self.logger.debug(f'VBV file {vbv}')
        return vbv
        # queue.put(vbv)
        # try:
        #     if not queue.empty():
        #         self.logger.debug("TEST : Queue length %s",queue.qsize())
        #     else:
        #         self.logger.debug("TEST : Empty queue in generate_vbv")
        # except Exception as ex :
        #     self.logger.debug("TEST : Error1 %s",ex)
    
    def duplicate_elements(self,arr, n):
        duplicated_array = []
        for element in arr:
            duplicated_array.extend([element] * n)
        return duplicated_array
    
    
    