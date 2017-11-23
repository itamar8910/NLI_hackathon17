import indicoio
import pickle
import cv2
from os.path import join
from os.path import isdir
from os import mkdir
import face_recognition
indicoio.config.api_key = '78b44753116b224e21ac1686033bdcb7'

def extract_faces_indico(img_path, dst_dir, sensitivity = 0.8):
    "returns list of face imgs extracted. note: not all imgaes contain faces, the identity recog should filter those"
    if not isdir(dst_dir):
        mkdir(dst_dir)
    im = cv2.imread(img_path)
    results = indicoio.facial_localization(img_path, sensitivity=sensitivity)
    gened_images = []
    for index in xrange(len((results))):
        faceim_name = 'face_' + str(index) + '.jpg'
        gened_images.append(faceim_name)
        cv2.imwrite(join(dst_dir, faceim_name), im[results[index]['top_left_corner'][1]:results[index]['bottom_right_corner'][1],
                                                results[index]['top_left_corner'][0]:results[index]['bottom_right_corner'][0]])
    return gened_images

def extract_faces_local(img_path, dst_dir, sensitivity = 0.8):
    "returns list of face imgs extracted. note: not all imgaes contain faces, the identity recog should filter those"
    if not isdir(dst_dir):
        mkdir(dst_dir)
    results = face_recognition.face_locations(face_recognition.load_image_file(img_path))
    print results
    im = cv2.imread(img_path)
    gened_paths = []

    for index in xrange(len(results)):
        faceim_name = 'face_' + str(index) + '.jpg'
        gened_paths.append(join(dst_dir, faceim_name))
        y , x, height, width = 0 , 0, 0, 0
        #(top, right, bottom, left)
        y = results[index][0]
        x = results[index][3]
        height = results[index][2] - results[index][0]
        width = results[index][1] - results[index][3]
        EXTEND = 100
        x -= EXTEND
        y -= EXTEND
        width += EXTEND*2
        height += EXTEND*2
        cv2.imwrite(join(dst_dir, faceim_name), im[y:y+height,x:x+width])
    return len(results), gened_paths

#    print face_recognition.face_locations(face_recognition.load_image_file("test_biden1.jpg"))


if __name__ == "__main__":
    extract_faces_local("/home/itamar/programming/nli/test_data/people.jpg", "poeple_faces2")
