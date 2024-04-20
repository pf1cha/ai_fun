from ultralytics import YOLO
import supervision as sv
import numpy as np

# Path for video
VIDEO_PATH = 'video_3.mp4'
# Path for result
SAVE_PATH = 'v9result' + VIDEO_PATH[6] + '.mp4'
# Load the model YOLOv9
model = YOLO('yolov9c.pt')
# Initialize all annotators
round_box_annotator = sv.RoundBoxAnnotator(
    color=sv.ColorPalette.ROBOFLOW,
    thickness=2
)
label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW)
trace_annotator = sv.TraceAnnotator()
tracker = sv.ByteTrack()

# Line for counting cars
LINE_START = sv.Point(200, 0)
LINE_END = sv.Point(200, 1280)
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
line_annotator = sv.LineZoneAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=0.5
)


def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model.track(frame, tracker="bytetrack.yaml")[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 2]
    detections = tracker.update_with_detections(detections)

    # make label for object
    labels = [
        f"#{tracker_id} {results.names[class_id]} {conf:0.2f}"
        for class_id, tracker_id, conf
        in zip(detections.class_id, detections.tracker_id, detections.confidence)]

    results_line = line_zone.trigger(detections=detections)
    annotated_frame = round_box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = line_annotator.annotate(frame=annotated_frame, line_counter=line_zone)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)

    return annotated_frame


sv.process_video(source_path=VIDEO_PATH, target_path=SAVE_PATH, callback=callback)
