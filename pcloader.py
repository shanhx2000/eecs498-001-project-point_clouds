import numpy as np
import random

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

def get_boundary(PC):
    """
    Return the upper and lower bound of each dimension.
    """
    return np.max(PC, axis=0), np.min(PC, axis=0)

def get_sparse_PC(PC, target_N):
    """
    Filter out concentrated data; Guarantee that the generated PC has the size of `target_N`.
    """
    ratio = 20
    upper_bound, lower_bound = get_boundary(PC)
    # Generate max, min, and diff for each dimension
    max_x, max_y, max_z = upper_bound[0], upper_bound[1], upper_bound[2]
    min_x, min_y, min_z = lower_bound[0], lower_bound[1], lower_bound[2]
    # diff is the side length of each small cube
    diff_x, diff_y, diff_z = (max_x-min_x) / ratio, (max_y-min_y) / ratio, (max_z-min_z) / ratio
    count = np.zeros((ratio, ratio, ratio))
    # point_dict records which cube one point belongs to
    point_dict = {}
    for i in range(PC.shape[0]):
        # Count the number of points in each block
        x = (int)((PC[i][0]-min_x) / diff_x)
        y = (int)((PC[i][1]-min_y) / diff_y)
        z = (int)((PC[i][2]-min_z) / diff_z)
        # print(x,y,z)
        x = min(ratio-1, x)
        y = min(ratio-1, y)
        z = min(ratio-1, z)
        count[x][y][z] += 1
        if str(x)+','+str(y)+','+str(z) in point_dict:
            point_dict[str(x)+','+str(y)+','+str(z)].append(i)
        else:
            point_dict[str(x)+','+str(y)+','+str(z)] = [i]
    count = np.ceil(count / PC.shape[0] * target_N)
    while (np.sum(count) != target_N):
        if np.sum(count) > target_N:
            # Pick one non-zero entry to reduce 1 point
            while True:
                rand_index = np.random.rand(3,1) * ratio
                if count[(int)(rand_index[0])][(int)(rand_index[1])][(int)(rand_index[2])] > 0:
                    count[(int)(rand_index[0])][(int)(rand_index[1])][(int)(rand_index[2])] -= 1
                    break
        else:
            # Pick one still available entry to add 1 point
            while True:
                rand_index = np.random.rand(3,1) * ratio
                x = (int)(rand_index[0])
                y = (int)(rand_index[1])
                z = (int)(rand_index[2])
                if str(x)+','+str(y)+','+str(z) in point_dict and count[x][y][z] < len(point_dict[str(x)+','+str(y)+','+str(z)]):
                    count[x][y][z] += 1
                    break
    target_PC = np.zeros((target_N, 3))
    index = 0
    for i in range(ratio):
        for j in range(ratio):
            for k in range(ratio):
                if count[i][j][k] == 0:
                    continue
                tmp_point_index_list = random.sample(point_dict[str(i)+','+str(j)+','+str(k)], int(count[i][j][k]))
                for point_index in tmp_point_index_list:
                    target_PC[index] = PC[point_index]
                    index += 1
    return target_PC


if __name__ == '__main__':
    data_dir = "./data/tutorials/"
    filename = "ism_train_cat.pcd"
    data = load_pcl(filename=(data_dir+filename))
    print(data.shape)
