import commands

import indicoio
import os
import json
from face_recog_local import recognize
from age_gender_estimation.get_age_gender import get_age_gender
from get_wiki_info import get_year_of_birth
from subprocess import Popen, PIPE, STDOUT
indicoio.config.api_key = '78b44753116b224e21ac1686033bdcb7'


#gets the number of objects
def findThings(thePath):
   #finds the objects in the photo
   theObj=indicoio.image_recognition(thePath, top_n=3)

   return theObj.keys()


#gets the filling from the photo
def findEmotion(thePath):
   #finds the result of the emotions
   theResult = indicoio.fer(thePath)

   #finds the biggest emotion
   maxI = max(theResult, key=theResult.get)  # Just use 'min' instead of 'max' for minimum.
   return maxI


def get_textual_desc(meta_data):
    inp = "Persons: "
    ordered_pds = []
    for pd in meta_data['persons']:
        if pd[0] != 'N/A':
            ordered_pds.append(pd)
    for pd in meta_data['persons']:
        if pd[0] == 'N/A':
            pd[0] = "*anonymous*"
            ordered_pds.append(pd)
    print ordered_pds
    for pd in ordered_pds:
        inp += pd[0] + " "
    inp += "- "
    inp += "Emotions: "
    for pd in ordered_pds:
        inp += pd[3] + " "
    inp += "- "
    inp += "Objects: "
    for obj in meta_data['objects']:
        inp += obj + " "
    inp += "- "
    print inp
    #p = Popen(['java', '-jar', './textGen.jar'], stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    #stdout, stderr = p.communicate(input=inp)
    #print stdout
    #return str(stdout)
    cmd = 'java -jar ./textGen.jar ' + inp
    print cmd
    res = commands.getstatusoutput(cmd)
    print "res"
    print res
    return " ".join(res[1].split("\n"))


def gen_metadata(img_path):
    """
    generates image metadata: persons in the images and their emotions, and the objects in the image.
    :param img_path:
    :return: a dict:{'persons':[['<name>','<emotion>'],...], 'objects':['<obj1>',...]}
    """
    model_path = "knn_train_1.p"
    names, paths = recognize(img_path, model_path)
    persons_md = []
    avg_pic_year = 0
    pic_year_count = 0
    for name, path in zip(names, paths):
        print name, path
        emotion = findEmotion(path)
        age, gender = get_age_gender(path, name = name)
        persons_md.append([name, int(age[0]), gender[0], emotion])
        person_yob = get_year_of_birth(name)
        if name != "N/A":
            avg_pic_year += person_yob + age
            pic_year_count += 1
    objects = findThings(img_path)
    meta_data = {'persons':persons_md, 'objects':objects, 'year':int(avg_pic_year/float(pic_year_count))}
    textual_desc = get_textual_desc(meta_data)
    meta_data['desc'] = textual_desc
    file_name = img_path if "/" not in img_path else img_path[img_path.rfind("/")+1:]
    meta_data['imgName'] = file_name
    return meta_data

def gen_json_metadata_for_dir(dir_path, json_dst_path):
    mds = []
    for img_path in os.listdir(dir_path):
        mds.append(gen_metadata(img_path))
    with open(json_dst_path,'w') as f:
        f.write(json.dumps(mds))

if __name__ == "__main__":

    gen_json_metadata_for_dir('tst_md_dir1_1','tst_md_dir1_1.json')
    exit()
    # inp = "Persons: biden *anonymous* - Emotions: Happy Happy - Objects: bow tie, bow-tie, bowtie Windsor tie groom, bridegroom suit, suit of clothes abaya - "
    # cmd = 'java -jar ./textGen.jar ' + inp
    # print cmd
    # res = commands.getstatusoutput(cmd)
    # #os.system(cmd + " > textGen.txt")
    # #print open('textGen.txt','w').read()
    # p = Popen(['java', '-jar', './textGen.jar', inp], stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    # stdout, stderr = p.communicate(input=inp)
    # for line in stdout:
    #     print line
    # print stdout
    # exit()
    #md = gen_metadata("/home/itamar/PycharmProjects/facedetection/obama_and_biden.jpg")
    md = gen_metadata("golda_test1.jpg")
    print md
    print "*****"
    print "People in image:"
    for pd in md['persons']:
        print "person md:" , pd
    print "Things in image:"
    for od in md['objects']:
        print od
