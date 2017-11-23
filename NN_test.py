import numpy as np
from sklearn import neighbors
import pickle

#X : [N_samples, N_feats]
#Y: [N_SAMPLES]
#PRED: [N_SAMPLES]

clf = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
N_FEATS = 512
randDataClassA = np.random.rand(100, N_FEATS)
randDataClassB = np.random.rand(100, N_FEATS)
yA = np.array(['a' for i in randDataClassA])
yB = np.array(['b' for i in randDataClassB])

clf.fit(np.concatenate((randDataClassA, randDataClassB)), np.concatenate((yA,yB)))
print "done"


print clf.predict(np.random.rand(1, N_FEATS))