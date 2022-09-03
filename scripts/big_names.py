import numpy as np
import matplotlib.pyplot as plt
import glob
import os

filename_list = glob.glob(os.path.join('cvpr_original', 'cvpr[0-9][0-9][0-9][0-9].txt'))

name_dict = {}

for filename in filename_list:
    with open(filename, 'r', encoding='utf8') as f:
        line_list = f.readlines()
        for line in line_list[1::4]:
            name_list = line.strip().split(", ")
            for name in name_list:
                name_dict[name] = name_dict.get(name, 0) + 1

name_dict = [item for item in name_dict.items() if item[1] > 20]
name_dict = sorted(name_dict, key=lambda item: item[1])
name_dict = name_dict[-20:]
name_list = [item[0] for item in name_dict]
count_list = [item[1] for item in name_dict]
y_pos = np.arange(len(name_list))
plt.barh(y_pos, count_list)
plt.yticks(y_pos, name_list)
plt.xlabel("count")
plt.title("Big Names (CVPR 2014-2022)")
plt.show()
