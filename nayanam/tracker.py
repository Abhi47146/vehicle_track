import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
from multiprocessing import Process, Queue
from configparser import ConfigParser
from datetime import datetime
import redis
import json
import copy
import traceback
from itspelogger import Logger
from nayanam.zone_assign import ZoneAssignment
from nayanam.line_count import LineVehicleCounter
import cv2
import time
import ast
from shapely.geometry import Polygon
from collections import defaultdict
from nayanam.queue_length import QueueLengthCalculator
from nayanam.vehicle_out_per_sec import VehicleOutCrossCounter
from nayanam.vehicle_in_per_sec import VehicleInCrossCounter
from nayanam.real_distance import RealDistance

def reconnect_to_camera(source_path, logger):
    while True:
        video_capture = cv2.VideoCapture(source_path)
        if video_capture.isOpened():
            logger.debug(f"Successfully reconnected to camera {source_path}")
            return video_capture
        else:
            logger.error(f"Failed to reconnect to camera. Retrying in 5 seconds... {source_path}")
            time.sleep(5)

def process_video(stream_id, source_path, target_path, arm_id, flag, zone_configfile, classes, custom_names, frame_queue, result_queue, fps, queue_check, detector, queue_detector, min_queue_length):
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
    logger = Logger('Tracker')
    try:
        parser = ConfigParser()
        parser.read(zone_configfile)
        zone_cordinates = parser.get(arm_id, 'zone_dimensions_1')
        line_cordinates = parser.get(f"{arm_id}_SPEED", 'line_segment_coordinates_1')
        list_of_zone_coordinates = [float(coord) for coord in zone_cordinates.split(',')]
        list_of_line_coordinates = [float(coord) for coord in line_cordinates.split(',')]
        zone_cord = [float(x) for x in zone_cordinates.split(',')]
        homography_pixel = np.array(zone_cord, dtype=np.float32).reshape(-1, 2)
        x1, y1, x2, y2 = map(int, list_of_line_coordinates)
        zone_obj = ZoneAssignment(arm_id, zone_configfile)
        video_capture = reconnect_to_camera(source_path, logger)
        
        #fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        fps = int(fps)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(target_path, fourcc, fps, (640, 640))
        tracking_ids = []
        tracker = sv.ByteTrack()  # Initialize ByteTracker for this stream
        counter = LineVehicleCounter(arm_id, line_cordinates)
        live_counter=0

        #Queue length setup
        coordinates = [
        (list_of_zone_coordinates[0], list_of_zone_coordinates[1]),
        (list_of_zone_coordinates[2], list_of_zone_coordinates[3]),
        (list_of_zone_coordinates[4], list_of_zone_coordinates[5]),
        (list_of_zone_coordinates[6], list_of_zone_coordinates[7])
        ]
        queue_coordinates = [
        (0, 0), (0, 640), (640, 640), (640, 0)
        ]
        queue_polygon=Polygon(queue_coordinates)
        queue_calculator = QueueLengthCalculator()
        queue_calculator.fps = fps
        queue_calculator.polygon = queue_polygon
        queue_calculator.vehicle_history = defaultdict(list)

        line_in_p1 = (int(float(list_of_zone_coordinates[4])), int(float(list_of_zone_coordinates[5])))
        line_in_p2 = (int(float(list_of_zone_coordinates[6])), int(float(list_of_zone_coordinates[7])))
        line_out_p1 = (int(float(list_of_zone_coordinates[0])), int(float(list_of_zone_coordinates[1])))
        line_out_p2 = (int(float(list_of_zone_coordinates[2])), int(float(list_of_zone_coordinates[3])))
        
        mid_in = ((line_in_p1[0]+line_in_p2[0])/2, (line_in_p1[1]+line_in_p2[1])/2)
        mid_out = ((line_out_p1[0]+line_out_p2[0])/2, (line_out_p1[1]+line_out_p2[1])/2)

        counter1 = VehicleOutCrossCounter(line_out_p1, line_out_p2, coordinates, fps)
        counter2 = VehicleInCrossCounter(line_in_p1, line_in_p2)
        real_world_cordinates = 0
        real_world = 0
        real_pixel = 0
        real_length = 0
        if queue_check:
            real_world_cordinates = parser.get(arm_id, 'real_world_points_1')
            real_world = [float(x) for x in real_world_cordinates.split(',')]
            real_pixel = np.array(real_world, dtype=np.float32).reshape(-1, 2)
            real_length = RealDistance(homography_pixel, real_pixel)

    except Exception as err:
        logger.error(f"Error: {err} | Traceback: {traceback.format_exc()}")
        return
    
    try:
        frame_count = 0
        dynamic_check = 0
        last_update = time.monotonic()
        while True:
            ret, frame = video_capture.read()
            if not ret:
                logger.error(f"Lost connection to camera {source_path}. Attempting to reconnect...")
                video_capture.release()
                video_capture = reconnect_to_camera(source_path, logger)
                continue
            
            frame = cv2.resize(frame, (640, 640))
            frame_queue.put((stream_id, frame))  # Send stream_id and frame to inference process
            
            stream_id_recv, detections = result_queue.get()  # Get inference results

            if stream_id_recv != stream_id:
                logger.error(f"Stream ID mismatch: expected {stream_id}, got {stream_id_recv}")
                continue
            
            # Update the tracker with the detections

            detections = tracker.update_with_detections(detections)
            tracked_objects = []
            for class_id, tracker_id, xyxy in zip(detections.class_id, detections.tracker_id, detections.xyxy):
                class_id = int(class_id)
                if class_id not in classes:
                    continue
                center_x = int((xyxy[0] + xyxy[2]) / 2)
                center_y = int((xyxy[1] + xyxy[3]) / 2)
                bottom_x = int((xyxy[0] + xyxy[2]) / 2)
                bottom_y = int(xyxy[3])
                upper_y = int(xyxy[1])
                centroid = (bottom_x, bottom_y, upper_y, center_x, center_y)
                crossed = counter.update(tracker_id, (bottom_x, bottom_y), y1)
                tracked_objects.append({'id': tracker_id, 'centroid': centroid, 'class_id':class_id})
                if tracker_id not in tracking_ids:
                    tracking_ids.append(tracker_id)
                if crossed == 1:
                    print("value of live count",live_counter)
                    live_counter+=1
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                    cv2.putText(frame, f"#{tracker_id} {custom_names[class_id]}", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    vbv_file = {
                        "TimeStamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")),
                        "SCN": arm_id,
                        "ObjectID": str(class_id),
                        "VehicleClass": custom_names[class_id],
                        "Event": 'vehicle-entry',
                        "Speed": str(-1)
                    }
                    logger.info(f"Generated vbv_file {vbv_file}")
                    redis_client.publish(detector, json.dumps(vbv_file))
                else:
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 255), 2)
                    cv2.putText(frame, f"#{tracker_id} {custom_names[class_id]}", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if len(tracking_ids) >= 1000:
                tracking_ids = tracking_ids[100:]
            if frame_count > 2400:
                frame_count = 0
            
            if queue_check:
                qlen = 0
                vout_ps=0
                counter1.update(tracked_objects, frame_count, custom_names)
                vehicles_in_zone = counter1.get_zone_last_second()
                vin_ps = counter2.update(tracked_objects)
                vout_ps = counter1.get_crossing_last_second()
                qlen = queue_calculator.update(tracked_objects, vout_ps, real_length)
                count_vehicle_queue = queue_calculator.get_vehicle_queue()
                start, end = queue_calculator.get_coordinates_queue()
                homography_pixel = real_length.calculate_homography()
                total_normalise_vehicles = counter1.normalise_vehicle()
                zone_length = real_length.calculate_real_world_distance(homography_pixel, mid_in, mid_out)
                start1 = (start[0],start[1])
                end1 = (end[0],end[1])

                cv2.putText(frame, f"Vehicle in Zone: {total_normalise_vehicles}", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Vehicles out per sec: {vout_ps}", (80, 80),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Vehicles in per sec: {vin_ps}", (20, 20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) 
                if time.monotonic() - last_update >= 1:
                    qpresent = 0
                    qtype = ""
                    density = total_normalise_vehicles / zone_length
                    if qlen:
                        if qlen >= min_queue_length:
                            qpresent = 1
                            qtype = "static_queue"
                            dynamic_check = 1
                        else:
                            start1 = (0,0)
                            end1 = (0,0)
                            qlen = 0
                    else:
                        if total_normalise_vehicles >= 4 and dynamic_check:
                            qpresent = 1
                            qtype = "dynamic_queue"
                            copy_objects = copy.deepcopy(tracked_objects)
                            qlen = queue_calculator.get_dynamic_queue_length(copy_objects)
                        else:
                            dynamic_check = 0
                    vbv_file = {
                            "TimeStamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")),
                            "SCN": arm_id,
                            "queue_present": qpresent,
                            "type_of_queue": qtype,
                            "queue_length": qlen,
                            "vehicle_in_per_second": vin_ps,
                            "vehicle_out_per_second": vout_ps,
                            "vehicle_in_zone": vehicles_in_zone,
                            "density_of_zone": density

                        }
                    logger.info(f"Generated vbv_file {vbv_file}")
                    redis_client.publish(queue_detector, json.dumps(vbv_file))
                    last_update = time.monotonic()   # reset timer
                cv2.putText(frame, f"Queue Length: {qlen}",  (110, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) 
                cv2.putText(frame, f"Vehicle in Queue: {count_vehicle_queue}", (130, 130),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.line(frame, start1, end1, (255, 255, 0), 2)
                cv2.putText(frame, f"Queue Type: {qtype}",  (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if flag:
                cv2.line(frame, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=2)
                cv2.putText(frame, f"Total Vehicles: {live_counter}", (160, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.line(frame, line_out_p1, line_out_p2, color=(0, 255, 255), thickness=2)
                cv2.line(frame, line_in_p1, line_in_p2, color=(0, 0, 255), thickness=2)
                video_writer.write(frame)
            
            # cv2.imshow("live tracking",frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            frame_count += 1

    except Exception as err:
        logger.error(f"Error: {err} | Traceback: {traceback.format_exc()}")
    finally:
        video_capture.release()
        video_writer.release()

def inference_worker(frame_queues, result_queues,device,model_id):
    
    model_filenames = {
        1: "bilaspur_best.pt",
        2: "dahod_best.pt",
        3: "bilaspur_best.engine",
        4: "bilaspur_best_openvino_model/",
        5: "dahod_best_openvino_model/"
    }
    #for production(LPU release uncomment this )
    model_path = os.path.join(os.path.dirname(__file__), model_filenames[model_id])

    #for local production(comment this while releasing)
    #model_path=f'home/itspe/Documents/nayanam-v2/nayanam_v3/nayanam/{model_filenames[model_id]}'
    print("model_path:",model_path)
    if device =='cuda':
        trt_model = YOLO(model_path).to(device)
    else:
        trt_model = YOLO(model_path)

    while True:
        for stream_id, frame_queue in frame_queues.items():
            if not frame_queue.empty():
                stream_id_recv, frame = frame_queue.get()  # Get stream_id and frame
                results = trt_model(frame)[0]
                detections = sv.Detections.from_ultralytics(results)
                result_queues[stream_id].put((stream_id_recv, detections))  # Send stream_id and detections back

def main():
    print(f"os.outputcount {os.cpu_count()}")
    conf = ConfigParser()
    conf.read('/home/abhishek/IOT/nayanam/config.ini')
    junction_scns = conf.get("Junctions", "scns").split(',')
    debug_flag = conf.getboolean("video_debug", "flag")
    device= conf.get("models", "device")
    print(f"device is {device}")
    model_id=conf.getint("models","model_id")
    classes = conf.get("classes", "id").split(',')
    fps = conf.get("frame_speed", "fps")
    queue_check = conf.getboolean("queue_detector", "queue_check")
    class_ids = [int(class_id) for class_id in classes]
    results_videos = conf.get("video_results", "pathnames").split(',')
    zone_configfile = conf.get("zone_config", "config")
    custom_names = ast.literal_eval(conf.get("class_name", "custom_names"))
    detector = conf.get("count_info", "channel")
    queue_detector = conf.get("queue_info", "channel")
    min_queue_length=conf.getint("queue_detector","min_queue_length")

    frame_queues = {}
    result_queues = {}
    processes = []

    for i in range(len(junction_scns)):
        frame_queues[i] = Queue()
        result_queues[i] = Queue()

    # Start the inference worker process
    inference_process = Process(target=inference_worker, args=(frame_queues, result_queues,device,model_id))
    inference_process.start()
    processes.append(inference_process)

    # Start the video processing processes
    for i in range(len(junction_scns)):
        video_stream = conf.get(junction_scns[i], 'stream')
        p = Process(target=process_video, args=(i, video_stream, results_videos[i], junction_scns[i], debug_flag, zone_configfile, class_ids, custom_names, frame_queues[i], result_queues[i], fps, queue_check, detector, queue_detector, min_queue_length))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()