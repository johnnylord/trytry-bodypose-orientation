import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.dirname((osp.abspath(__file__)))))

import argparse
import io
import cv2
import numpy as np
from PIL import Image

import grpc
import message.object_detection_pb2 as object_detection_pb2
import message.pose_estimation_pb2 as pose_estimation_pb2
import service.object_detection_pb2_grpc as object_detection_pb2_grpc
import service.pose_estimation_pb2_grpc as pose_estimation_pb2_grpc
from utils.convert import pil_to_bytes
from utils.assign import match_bbox_keypoint
from utils.transform import scale_bboxes_coord, scale_keypoints_coord


# Global environment
OBJECT_DETECTION_SERVER_IP = "140.112.18.214"
OBJECT_DETECTION_SERVER_PORT = 50002
POSE_ESTIMATION_SERVER_IP = "140.112.18.214"
POSE_ESTIMATION_SERVER_PORT = 50003
FRAME_SIZE = (512, 512)

def main(args):
    # Create dataset directory
    img_dir = osp.join(args['output'], 'img')
    pose_dir = osp.join(args['output'], 'pose')
    if not osp.exists(img_dir):
        os.makedirs(img_dir)
    if not osp.exists(pose_dir):
        os.makedirs(pose_dir)

    # Read video input
    cap = cv2.VideoCapture(args['video'])
    if not cap.isOpened():
        raise RuntimeError("Video '{}' cannot be opened".format(args['video']))

    # Establish service connection
    with \
        grpc.insecure_channel(f"{OBJECT_DETECTION_SERVER_IP}:{OBJECT_DETECTION_SERVER_PORT}") as detection_channel, \
        grpc.insecure_channel(f"{POSE_ESTIMATION_SERVER_IP}:{POSE_ESTIMATION_SERVER_PORT}") as estimation_channel \
    :
        # Create service stubs
        detection_service = object_detection_pb2_grpc.DetectionStub(detection_channel)
        estimation_service = pose_estimation_pb2_grpc.EstimationStub(estimation_channel)

        # Processing video frame by frame
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        count = fps
        next_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Process frame every second not every frame
            count -= 1
            if count > 0:
                continue

            # Convert frame to pil image
            resized = cv2.resize(frame, FRAME_SIZE)
            img = Image.fromarray(resized)

            # Perform object detection
            # ================================================================
            request = object_detection_pb2.DetectRequest()
            request.img.payload = pil_to_bytes(img)
            response = detection_service.DetectObjects(request)

            # Filter out person bbox
            bboxes = np.array([ (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.conf)
                                for bbox in response.bboxes
                                if (
                                    bbox.label == 'person'
                                    and bbox.ymax-bbox.ymin > 100
                                )])

            # Perform pose estimation
            # ================================================================
            request = pose_estimation_pb2.EstimateRequest()
            request.img.payload = pil_to_bytes(img)
            response = estimation_service.EstimatePoses(request)

            # Extract keypoints
            all_keypoints = []
            for pose in response.poseKeypoints:
                points = np.array([ (p.x, p.y, p.conf) for p in pose.points ])
                all_keypoints.append(points)
            all_keypoints = np.array(all_keypoints)

            # Transform keypoints and bboxes to same coordinate system
            # ================================================================
            if len(bboxes) > 0:
                bboxes = scale_bboxes_coord(bboxes,
                                            old_resolution=resized.shape[:2],
                                            new_resolution=frame.shape[:2])
            if len(all_keypoints) > 0:
                all_keypoints = np.array([
                                    scale_keypoints_coord(keypoints,
                                        old_resolution=resized.shape[:2],
                                        new_resolution=frame.shape[:2])
                                    for keypoints in all_keypoints ])

            # Match all_keypoints to bboxes
            # ================================================================
            bodyposes = match_bbox_keypoint(bboxes, all_keypoints)
            if len(bodyposes) <= 0:
                continue

            # Crop person image patch
            patches = []
            patch_keypoints = []
            for bodypose in bodyposes:
                bbox = bodypose[:5]
                keypoints = bodypose[5:].reshape(25, 3)
                xmin, ymin, xmax, ymax, conf = bbox.tolist()
                patch = frame[int(ymin):int(ymax), int(xmin):int(xmax), :]
                patches.append(patch)
                patch_keypoints.append(keypoints)

            # Export image patch & keypoints
            for patch, keypoints in zip(patches, patch_keypoints):
                if next_id >= args['max_samples']:
                    break
                img_path = osp.join(img_dir, "{:04d}.jpg".format(next_id))
                pose_path = osp.join(pose_dir, "{:04d}.npy".format(next_id))
                cv2.imwrite(img_path, patch)
                np.save(pose_path, keypoints)
                next_id += 1

            # Refill count
            count = fps
            if next_id >= args['max_samples']:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="input video")
    parser.add_argument("--max_samples", default=1000, help="input video")
    parser.add_argument("--output", default="output", help="output directory")

    args = vars(parser.parse_args())
    main(args)
