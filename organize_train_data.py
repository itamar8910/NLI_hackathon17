from os import listdir
from os.path import join
from os import rename

def organize(src_path, label, dst_path):
    index = 0
    for f in listdir(src_path):
        rename(join(src_path, f), join(dst_path, label + "_" + str(index)))
        index += 1
if __name__ == "__main__":
    organize("/home/itamar/Downloads/Google Images/trump", "trump", "training_biden_trump")