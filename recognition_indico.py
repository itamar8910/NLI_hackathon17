import numpy as np
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
import indicoio
from os import listdir
from os.path import join

indicoio.config.api_key = 'your api key'

def get_images_feats(images_paths):
    "returns list of np arrays of feats"
    results = indicoio.facial_features(images_paths)
    return [np.array(results[i]) for i in xrange(len(results))]

def recognize(img_path, model_path):
    """
    returns list of the names of the persons that appear in the image
    :param img_path:
    :return:
    """
    # TODO: the image may contain more than one person, impl face localization and decision threshold
    img_feats = get_images_feats([img_path])[0]
    clf = pickle.load(open(model_path,'rb'))
    pred = clf.predict([img_feats])
    return pred[0]


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
    if model == "knn":
        print "knn classifer"
        clf = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
    elif model == "svm":
        print "svm classifier"
        clf = SVC()
    elif model == "forest":
        print "forest classifier"
        clf = RandomForestClassifier()
    print("starts fitting on train data")
    clf.fit(X, y)
    print("finished fitting on train data")

    pickle.dump(clf, open(model_output,'wb'))
    return set(y)

if __name__ == "__main__":
    print train_on_faces("/home/itamar/PycharmProjects/facedetection/training_faces_dummy", "forest_dummy_biden_bieber.p",model="knn")
    print recognize("test_biden1.jpg", "forest_dummy_biden_bieber.p")
    print recognize("test_biden2.jpg", "forest_dummy_biden_bieber.p")
    print recognize("test_biden3.jpg", "forest_dummy_biden_bieber.p")
    print recognize("test_beiber1.jpg", "forest_dummy_biden_bieber.p")
    print recognize("test_bieber2.jpg", "forest_dummy_biden_bieber.p")
    print recognize("test_beiber3.jpg", "forest_dummy_biden_bieber.p")
