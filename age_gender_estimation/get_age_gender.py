import os
import cv2
import dlib
import numpy as np
import argparse
from wide_resnet import WideResNet

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.18-4.06.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args

def get_age_gender(im_path):
    args = get_args()
    depth = args.depth
    k = args.width
    weight_file = args.weight_file

    if not weight_file:
        weight_file = os.path.join("age_gender_estimation/pretrained_models", "weights.18-4.06.hdf5")

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    img = cv2.imread(im_path)
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(input_img)

    detected = detector(input_img, 1)
    faces = np.empty((len(detected), img_size, img_size, 3))

    for i, d in enumerate(detected):
        x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
        xw1 = max(int(x1 - 0.4 * w), 0)
        yw1 = max(int(y1 - 0.4 * h), 0)
        xw2 = min(int(x2 + 0.4 * w), img_w - 1)
        yw2 = min(int(y2 + 0.4 * h), img_h - 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
        faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

    if len(detected) > 0:
        # predict ages and genders of the detected faces
        results = model.predict(faces)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()
        #print predicted_ages
        #print predicted_genders
        genders = ['M' if x[1] > x[0] else 'F' for x in predicted_genders]
        return predicted_ages, genders

if __name__ == "__main__":
    print get_age_gender("../im2.jpg")