import os
import cv2
import imutils
import time
import traceback
from multiprocessing import Process, Queue
from configparser import ConfigParser
from datetime import datetime
import ast
from nayanam.zone_assign import ZoneAssignment
from nayanam.data_sender import send_to_redis, send_to_api
from nayanam.logger import Logger

logger = Logger('segmentation')
class CentroidTracker:
    def __init__(self):
        self.next_id = 1
        self.objects = {}   # id -> centroid
        self.first_y = {}
        self.counted = set()

    def update(self, detections):
        updated_objects = {}
        for cx, cy in detections:
            assigned = False
            for oid, (px, py) in self.objects.items():
                if abs(cx - px) < 20 and abs(cy - py) < 20:
                    updated_objects[oid] = (cx, cy)
                    assigned = True
                    break
            if not assigned:
                updated_objects[self.next_id] = (cx, cy)
                self.first_y[self.next_id] = cy
                self.next_id += 1
        self.objects = updated_objects
        return self.objects

def reconnect_to_camera(source_path, logger):
    while True:
        video_capture = cv2.VideoCapture(source_path)
        if video_capture.isOpened():
            logger.debug(f"Successfully reconnected to camera {source_path}")
            return video_capture
        else:
            logger.error(f"Failed to reconnect to camera. Retrying in 5 seconds... {source_path}")
            time.sleep(5)

def inference_worker(frame_queues, result_queues, config_params):
    
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    min_area = config_params['min_area']

    while True:
        for stream_id, frame_queue in frame_queues.items():
            if not frame_queue.empty():
                sid, frame = frame_queue.get()
                
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fg_mask = bg_subtractor.apply(frame_gray)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                fg_mask = cv2.dilate(fg_mask, kernel, iterations=4)

                contours = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(contours)
                detections = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append({'bbox': (x, y, w, h),'area': area,'centroid': (int(x + w / 2), int(y + h))})
                
                result_queues[stream_id].put((sid, detections))

def process_video(stream_id, source_path, target_path, arm_id, flag, zone_configfile, custom_names, frame_queue, result_queue, fps, config_params, transport, publish, api):

    try:
        parser = ConfigParser()
        parser.read(zone_configfile)
        zone_cordinates = parser.get(arm_id, 'zone_dimensions_1')
        line_cordinates = parser.get(f"{arm_id}_SPEED", 'line_segment_coordinates_1')
        list_of_zone_coordinates = [float(coord) for coord in zone_cordinates.split(',')]
        list_of_line_coordinates = [float(coord) for coord in line_cordinates.split(',')]
        x1, y1, x2, y2 = map(int, list_of_line_coordinates)
        zone_obj = ZoneAssignment(arm_id, zone_configfile)
        video_capture = reconnect_to_camera(source_path, logger)
        fps=int(fps)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(target_path, fourcc, fps, (640, 640))
        tracker = CentroidTracker()
        track_y = y1
        count_y = y1 + 5
        vehicle_count = 0
        area_2w = config_params['area_2w']
        area_4w = config_params['area_4w']

    except Exception as err:
        logger.error(f"Error: {err} | Traceback: {traceback.format_exc()}")
        return

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                logger.error(f"Lost connection to camera {source_path}. Attempting to reconnect...")
                video_capture.release()
                video_capture = reconnect_to_camera(source_path, logger)
                continue

            frame = cv2.resize(frame, (640, 640))
            frame_queue.put((stream_id, frame))
            
            # Get results from inference_worker
            sid_recv, raw_detections = result_queue.get()

            current_detections = []
            # Temporary storage to map area to IDs
            area_map = {} 

            for det in raw_detections:
                cx, cy = det['centroid']
                if cy > track_y:
                    current_detections.append((cx, cy))
                    area = det['area']
                    if area < area_2w:
                        class_id = 0
                    elif area < area_4w:
                        class_id = 1
                    else:
                        class_id = 2
                    area_map[(cx, cy)] = class_id

            objects = tracker.update(current_detections)

            for oid, (cx, cy) in objects.items():
                veh_type = area_map.get((cx, cy), "unknown")
                start_y = tracker.first_y.get(oid, cy)

                if start_y < count_y:
                    if oid not in tracker.counted and cy < count_y and cy > track_y:
                        vehicle_count += 1
                        tracker.counted.add(oid)
                        print("value of live count",vehicle_count)
                        if transport == "REDIS":
                            vbv_file = {
                            "TimeStamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                            "SCN": arm_id,
                            "ObjectID": str(class_id),
                            "VehicleClass": custom_names[class_id],
                            "Event": 'vehicle-entry',
                            "Speed": str(-1)
                            }
                            send_to_redis(vbv_file, publish)
                        else:
                            vbv_file = {
                            "timestamp":str(datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")),
                            "scn": arm_id,
                            "id": str(class_id),
                            "lane":"1",
                            "axles":"",
                            "class": str(class_id+1),
                            "speed": "0"
                            }
                            send_to_api(vbv_file, api)
                    
                    box_color = (0, 255, 0) if oid in tracker.counted else (255, 0, 0)
                    cv2.rectangle(frame, (cx-25, cy-25), (cx+25, cy+25), box_color, 2)
                    cv2.putText(frame, f"ID:{oid} {veh_type}", (cx, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Draw Lines
            cv2.line(frame, (x1, track_y), (x2, y2), (0, 0, 255), 2)
            cv2.line(frame, (x1, count_y), (x2, y2+5), (0, 255, 0), 2)
            cv2.putText(frame, f"Vehicle_Count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if flag:
                video_writer.write(frame)
            cv2.imshow("Vehicle Counting (MOG2 + ROI)", frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break
            
    except Exception as err:
        logger.error(f"Error: {err} | Traceback: {traceback.format_exc()}")
    finally:
        video_capture.release()
        video_writer.release()


def main():
    print(f"os.outputcount {os.cpu_count()}")
    conf = ConfigParser()
    conf.read('/home/abhishek/Pictures/IOT/nayanam/config.ini')
    junction_scns = conf.get("Junctions", "scns").split(',')
    debug_flag = conf.getboolean("video_debug", "flag")
    device= conf.get("models", "device")
    print(f"device is {device}")
    fps = conf.get("frame_speed", "fps")
    results_videos = conf.get("video_results", "pathnames").split(',')
    zone_configfile = conf.get("zone_config", "config")
    custom_names = ast.literal_eval(conf.get("class_name", "custom_names"))
    transport = conf.get("data_transport", "transport")
    publish = conf.get("count_publish", "publish")
    api = conf.get("api", "api_url")
    config_params = {
        'min_area': 750,
        'area_2w': 3000,
        'area_4w': 6000
    }

    frame_queues = {}
    result_queues = {}
    processes = []

    for i in range(len(junction_scns)):
        frame_queues[i] = Queue()
        result_queues[i] = Queue()

    # Start the inference worker process
    inference_process = Process(target=inference_worker, args=(frame_queues, result_queues,config_params))
    inference_process.start()
    processes.append(inference_process)

    # Start Video Processing Process
    for i in range(len(junction_scns)):
        video_stream = conf.get(junction_scns[i], 'stream')
        p = Process(target=process_video, args=(i, video_stream, results_videos[i], junction_scns[i], debug_flag, zone_configfile, custom_names, frame_queues[i], result_queues[i], fps,config_params, transport, publish, api))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()


