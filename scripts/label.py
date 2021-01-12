import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.dirname((osp.abspath(__file__)))))
import argparse

import cv2
import numpy as np
import pandas as pd

from utils.display import draw_bodypose25, draw_text, draw_bbox

WIN_SIZE = (760, 1080)

SEMANTIC_LABEL = {
    0: "North",
    1: "North East",
    2: "East",
    3: "South East",
    4: "South",
    5: "South West",
    6: "West",
    7: "North West",
    }

def trackbar_callback(value):
    pass

def main(args):
    # Check dataset validity
    dataset_dir = args['input']
    img_dir = osp.join(dataset_dir, 'img')
    pose_dir = osp.join(dataset_dir, 'pose')
    assert osp.exists(img_dir) and osp.exists(pose_dir)
    assert len(os.listdir(img_dir)) and len(os.listdir(pose_dir))

    # Create display window
    length = len(os.listdir(img_dir))
    cv2.namedWindow("Display", cv2.WINDOW_GUI_EXPANDED)
    cv2.createTrackbar("ID", "Display", 0, length, trackbar_callback)
    if args['hint']:
        cv2.namedWindow("Hint", cv2.WINDOW_GUI_EXPANDED)
        path = f'{osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "images/hint.png")}'
        hint = cv2.imread(path)
    # Show dataset
    labels = {}
    img_files = sorted([ osp.join(img_dir, f) for f in os.listdir(img_dir) ])
    pose_files = sorted([ osp.join(pose_dir, f) for f in os.listdir(pose_dir) ])
    prelabels = list(pd.read_csv(args['history'])['label']) \
                if args['history'] is not None else [None]*len(img_files)
    for ifile, pfile, label in zip(img_files, pose_files, prelabels):
        current_id = int(osp.basename(ifile).split(".")[0])
        img = cv2.imread(ifile)
        pose = np.load(pfile)
        # Resize image
        old_size = img.shape[:2][::-1]
        new_size = (WIN_SIZE[0]//2, WIN_SIZE[1])
        x_scale, y_scale = (np.array(new_size)/np.array(old_size)).tolist()
        # Draw image
        img = cv2.resize(img, new_size)
        pose[:, 0] *= x_scale
        pose[:, 1] *= y_scale
        canvas = np.zeros((new_size[1], new_size[0], 3), dtype=np.uint8)
        canvas = draw_bodypose25(canvas, pose, thickness=5)
        # Show label
        if label is not None:
            draw_text(img, f"{SEMANTIC_LABEL[label]}",
                    position=(0, 0),
                    fgcolor=(255, 255, 255),
                    fontScale=3)
            draw_text(canvas, f"{SEMANTIC_LABEL[label]}",
                    position=(0, 0),
                    fgcolor=(255, 255, 255),
                    fontScale=3)
        # Show image
        frame = np.hstack([img, canvas])
        cv2.imshow("Display", frame)
        cv2.setTrackbarPos("ID", "Display", current_id)
        if args['hint']:
            cv2.imshow("Hint", hint)
        # Key handler
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif 48 <= key <= 55:
            label = key - 48
            labels[osp.basename(ifile)] = label
            print(f"Label: {SEMANTIC_LABEL[label]}")
        elif key == 32:
            jump = True
        elif label is not None:
            labels[osp.basename(ifile)] = label

    # Export label data (orient.csv)
    df = pd.DataFrame({ 'filename': labels.keys(),
                        'label': labels.values() }, index=False)
    df.to_csv(args['output'])
    print("Save label data to '{}'".format(args['output']))

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="dataset directory")
    parser.add_argument("--output", default="orient.csv", help="output file")
    parser.add_argument("--history", help="prelabel orientation")
    parser.add_argument("--hint", action='store_true', help="show label hint")

    args = vars(parser.parse_args())
    main(args)
