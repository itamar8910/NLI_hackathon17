import indicoio
from face_recog_local import recognize
from age_gender_estimation.get_age_gender import get_age_gender
from get_wiki_info import get_year_of_birth
indicoio.config.api_key = '78b44753116b224e21ac1686033bdcb7'


#gets the number of objects
def findThings(thePath):
   #finds the objects in the photo
   theObj=indicoio.image_recognition(thePath, top_n=5)

   return theObj


#gets the filling from the photo
def findEmotion(thePath):
   #finds the result of the emotions
   theResult = indicoio.fer(thePath)

   #finds the biggest emotion
   maxI = max(theResult, key=theResult.get)  # Just use 'min' instead of 'max' for minimum.
   return maxI

def gen_metadata(img_path):
    """
    generates image metadata: persons in the images and their emotions, and the objects in the image.
    :param img_path:
    :return: a dict:{'persons':[['<name>','<emotion>'],...], 'objects':['<obj1>',...]}
    """
    model_path = "knn_dummy_biden_bieber.p"
    names, paths = recognize(img_path, model_path)
    persons_md = []
    for name, path in zip(names, paths):
        emotion = findEmotion(path)
        age, gender = get_age_gender(path)
        persons_md.append([name, age[0], gender[0], emotion])
    objects = findThings(img_path)
    return {'persons':persons_md, 'objects':objects}

if __name__ == "__main__":
    md = gen_metadata("/home/itamar/PycharmProjects/facedetection/obama_and_biden.jpg")
    print md
    print "*****"
    print "People in image:"
    for pd in md['persons']:
        print "person md:" , pd
    print "Things in image:"
    for od in md['objects']:
        print od
