import os
import shutil

# create folder named "gcommands"

path_train = "./gcommands/train"
path_valid = "./gcommands/valid"

# your path to the project folder from the root. for example
com_path = "C:/Users/Idan/PycharmProjects/ML_4/ML_5/"


def create(folder, kind, original):
    os.mkdir(folder)
    subdir = [x[0] for x in os.walk(original)]
    subdir = subdir[1:]
    for m_dir in subdir:
        num_file = 0
        the_dir = m_dir.split("\\")[1]
        new_place = folder + "/" + the_dir
        os.mkdir(new_place)
        for file in os.listdir(m_dir):
            if file == ".":
                continue
            if num_file > 100:
                num_file = 0
                break
            num_file += 1
            shutil.copyfile(com_path + "/gcommands/" + kind + "/" + the_dir + "/" + file,
                            com_path + new_place + "/" + file)


create("short_train", "train", path_train)
create("short_valid", "valid", path_valid)
