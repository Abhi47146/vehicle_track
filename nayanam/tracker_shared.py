import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
from multiprocessing import Process, Event
from configparser import ConfigParser
from datetime import datetime
import traceback
from nayanam.zone_assign import ZoneAssignment
from nayanam.line_count import LineVehicleCounter
from nayanam.data_sender import send_to_redis, send_to_api
from nayanam.logger import Logger
from nayanam.shared_stream import SharedFrameChannel, SharedResultChannel
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


def process_video(stream_id, source_path, target_path, arm_id, flag, zone_configfile,
                   classes, custom_names, frame_shm_name, result_shm_name,
                   frame_ready, result_ready, fps, transport, publish, api):
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

        fps = int(fps)
        print("value of frame_speed", fps)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(target_path, fourcc, fps, (640, 640))
        tracker = sv.ByteTrack()
        counter = LineVehicleCounter(arm_id, line_cordinates)
        live_counter = 0

        # Attach to the shared-memory blocks created by main()
        frame_channel = SharedFrameChannel(stream_id, create=False, shm_name=frame_shm_name)
        result_channel = SharedResultChannel(stream_id, create=False, shm_name=result_shm_name)

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

            # --- hand the frame to the inference process via shared memory ---
            frame_channel.write(frame)
            frame_ready.set()

            # --- wait for the inference process to publish results ---
            result_ready.wait()
            result_ready.clear()
            xyxy, confidence, class_id = result_channel.read()

            detections = sv.Detections(
                xyxy=xyxy.astype(np.float32),
                confidence=confidence.astype(np.float32),
                class_id=class_id.astype(int),
            )
            detections = tracker.update_with_detections(detections)

            for class_id_val, tracker_id, box in zip(detections.class_id, detections.tracker_id, detections.xyxy):
                class_id_val = int(class_id_val)
                if class_id_val not in classes:
                    continue
                bottom_x = int((box[0] + box[2]) / 2)
                bottom_y = int(box[3])
                crossed = counter.update(tracker_id, (bottom_x, bottom_y), y1)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
                cv2.putText(frame, f"#{tracker_id} {custom_names[class_id_val]}", (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                if crossed == 1:
                    print("value of live count", live_counter)
                    live_counter += 1
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                    cv2.putText(frame, f"#{tracker_id} {custom_names[class_id_val]}", (int(box[0]), int(box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    if transport == "REDIS":
                        vbv_file = {
                            "TimeStamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                            "SCN": arm_id,
                            "ObjectID": str(class_id_val),
                            "VehicleClass": custom_names[class_id_val],
                            "Event": 'vehicle-entry',
                            "Speed": str(-1)
                        }
                        send_to_redis(vbv_file, publish)
                    else:
                        vbv_file = {
                            "timestamp": str(datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")),
                            "scn": arm_id,
                            "id": str(tracker_id),
                            "lane": "1",
                            "axles": "",
                            "class": str(class_id_val + 1),
                            "speed": "0"
                        }
                        send_to_api(vbv_file, api)

            if flag:
                video_writer.write(frame)

    except Exception as err:
        logger.error(f"Error: {err} | Traceback: {traceback.format_exc()}")
    finally:
        video_capture.release()
        video_writer.release()
        frame_channel.close()
        result_channel.close()


def inference_worker(stream_channels, device, model_id, yolo, openvino, engine, ncnn,
                      max_batch_size=8, batch_wait_s=0.005):
    
    model_filenames = {
        1: yolo,
        2: engine,
        3: openvino,
        4: ncnn
    }
    model_path = os.path.join(os.path.dirname(__file__), model_filenames[model_id])

    print("model_path:", model_path)
    if device == 'cuda':
        trt_model = YOLO(model_path).to(device)
    else:
        trt_model = YOLO(model_path)

    # Attach (not create) to every stream's shared-memory blocks in this process
    local = {}
    for stream_id, info in stream_channels.items():
        local[stream_id] = {
            "frame_channel": SharedFrameChannel(stream_id, create=False, shm_name=info["frame_shm_name"]),
            "result_channel": SharedResultChannel(stream_id, create=False, shm_name=info["result_shm_name"],
                                                   max_det=info["max_det"]),
            "frame_ready": info["frame_ready"],
            "result_ready": info["result_ready"],
        }

    num_streams = len(local)
    
    effective_batch_wait_s = batch_wait_s if num_streams > 1 else 0.0

    while True:
        ready_ids = [sid for sid, ch in local.items() if ch["frame_ready"].is_set()]

        if not ready_ids:
            time.sleep(0.001)
            continue

       
        if effective_batch_wait_s and len(ready_ids) < max_batch_size:
            deadline = time.time() + effective_batch_wait_s
            while len(ready_ids) < max_batch_size and time.time() < deadline:
                time.sleep(0.0005)
                ready_ids = [sid for sid, ch in local.items() if ch["frame_ready"].is_set()]

        batch_ids = ready_ids[:max_batch_size]

        frames = []
        for sid in batch_ids:
            ch = local[sid]
            ch["frame_ready"].clear()
            frames.append(ch["frame_channel"].read())

        # Batched call: list in, list of per-image Results out, same order.
        batch_results = trt_model(frames) if len(frames) > 1 else [trt_model(frames[0])[0]]

        for sid, results in zip(batch_ids, batch_results):
            detections = sv.Detections.from_ultralytics(results)
            ch = local[sid]
            ch["result_channel"].write(detections.xyxy, detections.confidence, detections.class_id)
            ch["result_ready"].set()


def main():
    print(f"os.outputcount {os.cpu_count()}")
    conf = ConfigParser()
    conf.read('/mnt/c/Documents and Settings/MANJU/Downloads/nayanam/nayanam/config.ini')
    junction_scns = conf.get("Junctions", "scns").split(',')
    debug_flag = conf.getboolean("video_debug", "flag")
    device = conf.get("models", "device")
    print(f"device is {device}")
    model_id = conf.getint("models", "model_id")
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

    max_det = conf.getint("models", "max_detections", fallback=128)
    max_batch_size = conf.getint("models", "max_batch_size", fallback=8)
    batch_wait_s = conf.getfloat("models", "batch_wait_seconds", fallback=0.005)

    processes = []
    stream_channels = {}          # passed to the inference process
    owned_channels = []           # kept alive in main() so blocks aren't GC'd; unlinked at the end

    for i in range(len(junction_scns)):
        frame_channel = SharedFrameChannel(i, create=True)
        result_channel = SharedResultChannel(i, create=True, max_det=max_det)
        frame_ready = Event()
        result_ready = Event()

        owned_channels.append((frame_channel, result_channel))
        stream_channels[i] = {
            "frame_shm_name": frame_channel.name,
            "result_shm_name": result_channel.name,
            "frame_ready": frame_ready,
            "result_ready": result_ready,
            "max_det": max_det,
        }

    # Single inference process serving every camera
    inference_process = Process(
        target=inference_worker,
        args=(stream_channels, device, model_id, yolo, openvino, engine, ncnn,
              max_batch_size, batch_wait_s)
    )
    inference_process.start()
    processes.append(inference_process)

    # One capture/tracking process per camera
    for i in range(len(junction_scns)):
        video_stream = conf.get(junction_scns[i], 'stream')
        info = stream_channels[i]
        p = Process(
            target=process_video,
            args=(i, video_stream, results_videos[i], junction_scns[i], debug_flag, zone_configfile,
                  class_ids, custom_names, info["frame_shm_name"], info["result_shm_name"],
                  info["frame_ready"], info["result_ready"], fps, transport, publish, api)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Only main() unlinks -- every other process only ever calls close()
    for frame_channel, result_channel in owned_channels:
        frame_channel.close()
        frame_channel.unlink()
        result_channel.close()
        result_channel.unlink()


if __name__ == "__main__":
    main()