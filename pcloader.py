import numpy as np

def load_pcl(filename="", st_idx=11):
    assert filename != ""
    ret = []
    with open(filename) as f:
        lines = f.readlines()
        lines = lines[st_idx:]
        for line in lines:
            words = line.split()
            ret.append([eval(words[0]),eval(words[1]),eval(words[2])])
    return np.array(ret)

if __name__ == '__main__':
    data_dir = "./data/tutorials/"
    filename = "ism_train_cat.pcd"
    data = load_pcl(filename=(data_dir+filename))
    print(data.shape)
