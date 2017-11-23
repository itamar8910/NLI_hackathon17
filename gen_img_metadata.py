import indicoio
from face_recog_local import recognize

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
