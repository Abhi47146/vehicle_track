import os
import supervision as sv
from ultralytics import YOLO
from multiprocessing import Process, Queue
from configparser import ConfigParser
from datetime import datetime
import traceback
from nayanam.zone_assign import ZoneAssignment
from nayanam.line_count import LineVehicleCounter
from nayanam.data_sender import send_to_redis, send_to_api
from nayanam.logger import Logger
import cv2
import time
import ast

logger = Logger('tracker')
def reconnect_to_camera(source_path, logger):
    while True:
        video_capture = cv2.VideoCapture(source_path)
        if video_capture.isOpened():
            logger.debug(f"Successfully reconnected to camera {source_path}")
            return video_capture
        else:
            logger.error(f"Failed to reconnect to camera. Retrying in 5 seconds... {source_path}")
            time.sleep(5)

def process_video(stream_id, source_path, target_path, arm_id, flag, zone_configfile, classes, custom_names, frame_queue, result_queue, fps, transport, publish, api):
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
        
        #fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        fps=int(fps)
        print("value of frame_speed", fps)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(target_path, fourcc, fps, (640, 640))
        tracker = sv.ByteTrack()  # Initialize ByteTracker for this stream
        counter = LineVehicleCounter(arm_id, line_cordinates)
        live_counter=0


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
            frame_queue.put((stream_id, frame))  # Send stream_id and frame to inference process
            
            stream_id_recv, detections = result_queue.get()  # Get inference results

            if stream_id_recv != stream_id:
                logger.error(f"Stream ID mismatch: expected {stream_id}, got {stream_id_recv}")
                continue
            

            detections = tracker.update_with_detections(detections)
            for class_id, tracker_id, xyxy in zip(detections.class_id, detections.tracker_id, detections.xyxy):
                class_id = int(class_id)
                if class_id not in classes:
                    continue
                bottom_x = int((xyxy[0] + xyxy[2]) / 2)
                bottom_y = int(xyxy[3])
                crossed = counter.update(tracker_id, (bottom_x, bottom_y), y1)
               # cv2.line(frame, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=2)
               # cv2.putText(frame, f"Total Vehicles: {live_counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 255), 2)
                cv2.putText(frame, f"#{tracker_id} {custom_names[class_id]}", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                if crossed == 1:
                    print("value of live count",live_counter)
                    live_counter+=1
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                    cv2.putText(frame, f"#{tracker_id} {custom_names[class_id]}", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
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
                        "id": str(tracker_id),
                        "lane":"1",
                        "axles":"",
                        "class": str(class_id+1),
                        "speed": "0"
                        }
                        send_to_api(vbv_file, api)
                
            if flag:
                video_writer.write(frame)

           # cv2.imshow("Vehicle Counting (MOG2 + ROI)", frame)
           # if cv2.waitKey(30) & 0xFF == 27:
            #    break

    except Exception as err:
        logger.error(f"Error: {err} | Traceback: {traceback.format_exc()}")
    finally:
        video_capture.release()
        video_writer.release()

def inference_worker(frame_queues,result_queues,device,model_id,yolo,openvino,engine,ncnn):
    
    model_filenames = {
        1: yolo,
        2: engine,
        3: openvino,
        4: ncnn
    }
    model_path = os.path.join(os.path.dirname(__file__), model_filenames[model_id])

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
    conf.read('/mnt/c/Documents and Settings/MANJU/Downloads/nayanam/nayanam/config.ini')
    junction_scns = conf.get("Junctions", "scns").split(',')
    debug_flag = conf.getboolean("video_debug", "flag")
    device= conf.get("models", "device")
    print(f"device is {device}")
    model_id=conf.getint("models","model_id")
    classes = conf.get("classes", "id").split(',')
    fps = conf.get("frame_speed", "fps")
    class_ids = [int(class_id) for class_id in classes]
    results_videos = conf.get("video_results", "pathnames").split(',')
    zone_configfile = conf.get("zone_config", "config")
    custom_names = ast.literal_eval(conf.get("class_name", "custom_names"))
    transport = conf.get("data_transport", "transport")
    publish = conf.get("count_publish", "publish")
    api = conf.get("api", "api_url")
    yolo = conf.get("yolo_pt", "pt")
    openvino = conf.get("yolo_openvino", "openvino")
    engine = conf.get("yolo_engine", "engine")
    ncnn = conf.get("yolo_ncnn", "ncnn")
    
    frame_queues = {}
    result_queues = {}
    processes = []

    for i in range(len(junction_scns)):
        frame_queues[i] = Queue()
        result_queues[i] = Queue()

    # Start the inference worker process
    inference_process = Process(target=inference_worker, args=(frame_queues,result_queues,device,model_id,yolo,openvino,engine,ncnn))
    inference_process.start()
    processes.append(inference_process)

    # Start the video processing processes
    for i in range(len(junction_scns)):
        video_stream = conf.get(junction_scns[i], 'stream')
        p = Process(target=process_video, args=(i, video_stream, results_videos[i], junction_scns[i], debug_flag, zone_configfile, class_ids, custom_names, frame_queues[i], result_queues[i], fps, transport, publish, api))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
