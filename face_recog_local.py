import numpy as np
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
from os import listdir, mkdir, remove
from os.path import join, isdir, basename, splitext
import face_recognition
import time
import os
from localiation import extract_faces_local


def get_images_feats(images_paths):
    "returns list of np arrays of feats"
    all_feats = []
    for im_path in images_paths:
        im = face_recognition.load_image_file(im_path)
        feats = face_recognition.face_encodings(im)[0] # assuming only one face in image
        all_feats.append(feats)
    return all_feats

def recognize(img_path, model_path):
    """
    returns list of the names of the persons that appear in the image
    :param img_path:
    :return:
    """

    # TODO: the image may contain more than one person, impl face localization and decision threshold

    cropped_dst_dir = join(os.path.dirname(os.path.realpath(__file__)),
                           "cropped_"+splitext(basename(img_path))[0])
    if not isdir(cropped_dst_dir):
        mkdir(cropped_dst_dir)


    num_faces, paths = extract_faces_local(img_path, cropped_dst_dir)
    if num_faces == 0:
        return None
    if num_faces == 1:
        paths = [img_path]
    imgs_feats = get_images_feats(paths)
    clf = pickle.load(open(model_path,'rb'))
    preds = clf.predict(imgs_feats)
    return preds


def get_face_label(name):
    return name[:name.index("_")].lower()


def train_on_faces(faces_dir, model_output, model = "knn"):
    """
    trains model on given faces and then pickles it in given dir.
    the imgs in faces_dir should be cropped images of faces,
    the part before the _ in the name of each image is its label, for example: rabin_xyz.jpg
    :param faces_dir:
    :param model_output:
    :return: labels of faces
    """
    X = []
    y = []
    train_paths = []
    for f in listdir(faces_dir):
        train_paths.append(join(faces_dir, f))
        y.append(get_face_label(f))
    X = get_images_feats(train_paths)
    clf = None
    if model == "knn":
        print "knn classifer"
        clf = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
    elif model == "svm":
        print "svm classifier"
        clf = SVC()
    elif model == "forest":
        print "forest classifier"
        clf = RandomForestClassifier()
    else:
        raise Exception("unsupported model type:" + str(model))
    print("starts fitting on train data")
    clf.fit(X, y)
    print("finished fitting on train data")

    pickle.dump(clf, open(model_output,'wb'))
    return set(y)

if __name__ == "__main__":
    pass
   # print face_recognition.face_locations(face_recognition.load_image_file("test_biden1.jpg"))
    #t = time.time()
    #print train_on_faces("/home/itamar/PycharmProjects/facedetection/training_faces_dummy", "forest_dummy_biden_bieber.p",model="knn")
    #print "training:" , time.time() - t
    #t = time.time()


    # print recognize("test_biden1.jpg", "forest_dummy_biden_bieber.p")
    # #print "query:" , time.time() - t
    # print recognize("test_biden2.jpg", "forest_dummy_biden_bieber.p")
    # print recognize("test_biden3.jpg", "forest_dummy_biden_bieber.p")
    # print recognize("test_beiber1.jpg", "forest_dummy_biden_bieber.p")
    # print recognize("test_bieber2.jpg", "forest_dummy_biden_bieber.p")
    # print recognize("test_beiber3.jpg", "forest_dummy_biden_bieber.p")