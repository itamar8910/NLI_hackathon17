# -*- coding: utf-8 -*-

import commands
import string
import indicoio
import os
import json
from face_recog_local import recognize
from age_gender_estimation.get_age_gender import get_age_gender
from get_wiki_info import get_year_of_birth
from googletrans import Translator

from subprocess import Popen, PIPE, STDOUT
indicoio.config.api_key = 'your api key'


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

ENG_TO_HEB_DICT = {
    'golda':u"גולדה",
    'rabin':u"רבין",
    'begin':u"בגין",
    'gurion':u"בן גוריון",
    'dayan':u"משה דיין",
    'Angry':u"כועס",
    'Happy':u"שמח",
    'Sad':u"עצוב",
    'Neutral':u"ניטרלי",
    'Surprise':u"מופתע",
    'female':u"נקבה",
    'male':u"זכר",
}

def translate_word_eng_to_heb(word):
    if word in ENG_TO_HEB_DICT.keys():
        return ENG_TO_HEB_DICT[word]
    else:
        return u"שגיאת תרגום"

def translate_sentence_eng_to_heb(sen):
    translator = Translator(service_urls=[
         'translate.google.com',
         'translate.google.co.kr',
       ])
    d = translator.translate(sen,src='en',dest='iw').text
    # printable = set(string.printable)
    # filter(lambda x: x not in printable, d)
    d = ''.join([i if not(ord(i) < 128) else ' ' for i in d])
    d = d.strip()
    return d


def translate_mds_to_heb(mds):
    for i in xrange(len(mds)):
        md = mds[i]
        for j in range(len(md['persons'])):
            pd = md['persons'][j]
            pd[0] = translate_word_eng_to_heb(pd[0])
            pd[2] = translate_word_eng_to_heb(pd[2])
            pd[3] = translate_word_eng_to_heb(pd[3])
            md['persons'][j] = pd
        md['desc'] = translate_sentence_eng_to_heb(md['desc'])
        for j in range(len(md['objects'])):
            od = md['objects'][j]
            md['objects'][j] = translate_sentence_eng_to_heb(od)
    return mds

def gen_json_metadata_for_dir(dir_path, json_dst_path):
    mds = []
    for img_path in os.listdir(dir_path):
        mds.append(gen_metadata(img_path))
    mds = translate_mds_to_heb(mds)
    with open(json_dst_path,'w') as f:
        f.write(json.dumps(mds))

if __name__ == "__main__":
    #print translate_sentence_eng_to_heb("hello gadarugzaq man")
    #exit()
    gen_json_metadata_for_dir('tst_md_dir1_1','tst_md_dir1_4.json')
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
