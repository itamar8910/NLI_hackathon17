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

def prep_data_from_dirs(pdir_path):
    datadir = join(pdir_path, "data")
    if not isdir(datadir):
        mkdir(datadir)
    for f in os.listdir(pdir_path):
        if f == "data":
            continue
        if isdir(join(pdir_path, f)):
            label = f
            for img in os.listdir(join(pdir_path, f)):
                new_name = label + "_" + img
                os.rename(join(pdir_path, f, img), join(datadir, new_name))


def get_images_feats(images_paths):
    "returns list of np arrays of feats"
    all_feats = []
    paths_not_found = []
    for im_path in images_paths:
        print "extracting face feats for:" , im_path
        im = face_recognition.load_image_file(im_path)
        feats_list = face_recognition.face_encodings(im)
        if len(feats_list) == 0:
            print "*** didn't find a face ***"
            paths_not_found.append(im_path)
            continue
        feats = feats_list[0] # assuming only one face in image
        all_feats.append(feats)
    return all_feats, paths_not_found

def recognize(img_path, model_path):
    """
    returns list[names of the persons that appear in the image, cropped_face_path]
    :param img_path:
    :return:
    """

    cropped_dst_dir = join(os.path.dirname(os.path.realpath(__file__)),
                           "cropped_"+splitext(basename(img_path))[0])
    if not isdir(cropped_dst_dir):
        mkdir(cropped_dst_dir)


    num_faces, paths = extract_faces_local(img_path, cropped_dst_dir)
    if num_faces == 0:
        return None
    if num_faces == 1:
        paths = [img_path]
    imgs_feats, _ = get_images_feats(paths)
    clf = pickle.load(open(model_path,'rb'))
    closest_distances = clf.kneighbors(imgs_feats, n_neighbors = 1)
    print closest_distances
    print paths
    DIST_THRESH = .5 # TODO: decide later based on expirs.
    not_recognized_i = [i for i in xrange(len(imgs_feats)) if closest_distances[0][i][0] > DIST_THRESH]

    preds = clf.predict(imgs_feats)
    for i in not_recognized_i:
        preds[i] = "N/A"
    return preds, paths


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
        print f
        train_paths.append(join(faces_dir, f))

    X, not_found = get_images_feats(train_paths)
    for p in train_paths:
        if p in not_found:
            print "not generating label to:", p
            continue
        label = get_face_label(p[p.rfind("/")+1:])
        print p, label
        y.append(label)
    print set(y)
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
   # print train_on_faces("/home/itamar/PycharmProjects/facedetection/train_data_1/photos/data",
    #                     "knn_train_1.p", model="knn")
    #"/home/itamar/PycharmProjects/facedetection/train_data_1/photos/data/begin_fsaf.jpg
    #print recognize("/home/itamar/PycharmProjects/facedetection/train_data_1/photos/data/golda_987fa.jpg",
     #               "knn_train_1.p")[0]
    print recognize("golda_test1.jpg",
                    "knn_train_1.p")[0]

    exit()
    #print recognize("biden_inaug.jpg", "knn_dummy_biden_bieber.p")

    #prep_data_from_dirs("/home/itamar/PycharmProjects/facedetection/train_data_1/photos")
    #exit()
    print train_on_faces("/home/itamar/PycharmProjects/facedetection/training_faces_dummy",
                         "knn_dummy_biden_bieber.p", model="knn")
    print recognize("biden_inaug.jpg", "knn_dummy_biden_bieber.p")
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