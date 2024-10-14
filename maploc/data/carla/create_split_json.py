import json
import random
import os, glob, shutil
from pathlib import Path
from tqdm import tqdm
import re 

from .dataset import CarlaDataModule
from ... import repo_dir
data_dir = CarlaDataModule.default_cfg["data_dir"]

# data_dir = "/home/rayguan/scratch1/carla/OrienterNet_ours/datasets/carla"
# repo_dir = "/home/rayguan/scratch1/carla/OrienterNet_ours"

saved_json_name = "splits_carla_loc"

SEED = 0
random.seed(SEED)

train_sample_interval = 3
# train_overhead_count = 10
train_overhead_count = 5
val_sample_interval = test_sample_interval = 10
# val_overhead_count = test_overhead_count = 3
val_overhead_count = test_overhead_count = 2
test_sample_ratio = 0.1

# train_min_route_len = 50
train_min_route_len = 80

# train Town1 Town2 Town4 Town5 Town10
# val Town1 Town2 Town4 Town5 Town10
# test Twon3 Town6 Town7 Town10

town_train = town_val = ['Town01', 'Town02', 'Town04', 'Town05', 'Town10HD']
town_test = ['Town03', 'Town06', 'Town07', 'Town10HD']



# scene_lst = [s for s in glob.glob(os.path.join(data_dir, "*")) if ".json" not in s ]
scene_lst = [s for s in os.listdir(data_dir) if ".json" not in s ]

train_scene = []
test_scene = []

for s in scene_lst:
    if s.split("_")[0] in town_train:
        train_scene.append(s)
    if s.split("_")[0] in town_test:
        test_scene.append(s)

# from IPython import embed; embed()

# train

train_route_lst = []

for s in train_scene:
    # train_route_lst.extend(os.listdir(os.path.join(data_dir, s)))
    train_route_lst.extend(glob.glob(os.path.join(data_dir, s, "*")))

# print(train_route_lst)
# print(len(train_route_lst))

filtered_train_route_lst = [r for r in train_route_lst if len(os.listdir(os.path.join(r, "lidar"))) > train_min_route_len]
# filtered_rest_train_route_lst = [r for r in train_route_lst if len(os.listdir(os.path.join(r, "lidar"))) <= train_min_route_len]

final_train_route = filtered_train_route_lst
final_train_route.sort()

print("Number of training routes:")
print(len(final_train_route))

final_train_data = []

for final_route in final_train_route:
    overhead_lst = os.listdir(os.path.join(final_route, "topdown_rgb"))
    overhead_lst.sort()
    if len(overhead_lst[1:]) <= train_overhead_count:
        overhead_lst = overhead_lst[1:]
    else:
        overhead_lst = random.sample(overhead_lst[1:], train_overhead_count)

    lidar_lst = os.listdir(os.path.join(final_route, "lidar"))
    lidar_lst.sort()

    for overhead in overhead_lst:
        start_idx = int(overhead.split(".")[0])
        start_idx = start_idx % train_sample_interval

        lidar_sample = lidar_lst[start_idx::train_sample_interval]
        for lidar in lidar_sample:
            final_train_data.append((final_route, 
                                    os.path.join("topdown_rgb", overhead), 
                                    lidar.split(".")[0]))

print("Number of training pairs:")
print(len(final_train_data))

# val

filtered_val_route_lst = [r for r in train_route_lst if len(os.listdir(os.path.join(r, "lidar"))) <= train_min_route_len]

final_val_route = []

for t in town_val:
    tmp_lst = [r for r in filtered_val_route_lst if t in r]
    sample_size = int(0.1 * len(tmp_lst))
    final_val_route.extend(random.sample(tmp_lst, sample_size))

final_val_route.sort()

print("Number of val routes:")
print(len(final_val_route))


final_val_data = []

for final_route in final_val_route:
    overhead_lst = os.listdir(os.path.join(final_route, "topdown_rgb"))
    overhead_lst.sort()
    overhead_lst = random.sample(overhead_lst[1:], val_overhead_count)

    lidar_lst = os.listdir(os.path.join(final_route, "lidar"))
    lidar_lst.sort()

    for overhead in overhead_lst:
        start_idx = int(overhead.split(".")[0])
        start_idx = start_idx % val_sample_interval

        lidar_sample = lidar_lst[start_idx::val_sample_interval]
        for lidar in lidar_sample:
            final_val_data.append((final_route, 
                                    os.path.join("topdown_rgb", overhead), 
                                    lidar.split(".")[0]))

print("Number of val pairs:")
print(len(final_val_data))



# test
test_route_lst = []

for s in test_scene:
    # train_route_lst.extend(os.listdir(os.path.join(data_dir, s)))
    test_route_lst.extend(glob.glob(os.path.join(data_dir, s, "*")))

# print(test_route_lst)
# print(len(test_route_lst))

filtered_test_route_lst = [r for r in test_route_lst if r not in filtered_train_route_lst]

final_test_route = []

for t in town_test:
    tmp_lst = [r for r in test_route_lst if t in r]
    sample_size = int(0.1 * len(tmp_lst))
    final_test_route.extend(random.sample(tmp_lst, sample_size))

final_test_route.sort()

print("Number of testing routes:")
print(len(final_test_route))



final_test_data = []

for final_route in final_val_route:
    overhead_lst = os.listdir(os.path.join(final_route, "topdown_rgb"))
    overhead_lst.sort()
    overhead_lst = random.sample(overhead_lst[1:], test_overhead_count)

    lidar_lst = os.listdir(os.path.join(final_route, "lidar"))
    lidar_lst.sort()

    for overhead in overhead_lst:
        start_idx = int(overhead.split(".")[0])
        start_idx = start_idx % test_sample_interval

        lidar_sample = lidar_lst[start_idx::test_sample_interval]
        for lidar in lidar_sample:
            final_test_data.append((final_route, 
                                    os.path.join("topdown_rgb", overhead), 
                                    lidar.split(".")[0]))

print("Number of test pairs:")
print(len(final_test_data))



# collect and save as json

split = {"train": {}, "val": {}, "test": {}, "test_shifts": {}}

for train in final_train_data:
    data_path, overhead_path, lidar_path = train
    data_path = "/".join(data_path.split("/")[-2:])
    if data_path not in split["train"]:
        split["train"][data_path] = {} 
    if overhead_path not in split["train"][data_path]:
        split["train"][data_path][overhead_path] = [] 

    split["train"][data_path][overhead_path].append(lidar_path)


for val in final_val_data:
    data_path, overhead_path, lidar_path = val
    data_path = "/".join(data_path.split("/")[-2:])
    if data_path not in split["val"]:
        split["val"][data_path] = {} 
    if overhead_path not in split["val"][data_path]:
        split["val"][data_path][overhead_path] = [] 

    split["val"][data_path][overhead_path].append(lidar_path)


for test in final_test_data:
    data_path, overhead_path, lidar_path = test
    data_path = "/".join(data_path.split("/")[-2:])
    if data_path not in split["test"]:
        split["test"][data_path] = {}
        split["test_shifts"][data_path] = {}
    if overhead_path not in split["test"][data_path]:
        split["test"][data_path][overhead_path] = [] 
        split["test_shifts"][data_path][overhead_path] = {}

    split["test"][data_path][overhead_path].append(lidar_path)
    split["test_shifts"][data_path][overhead_path][lidar_path] = [
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ]

# print(split)


with open(os.path.join(repo_dir, "maploc", "data", "carla", saved_json_name + ".json"), "w") as f:
    json.dump(split, f, indent=2)

print("Split file saved to " + os.path.join(repo_dir, "maploc", "data", "carla", saved_json_name + ".json"))

from datetime import date
today = date.today()

with open(os.path.join(repo_dir, "maploc", "data", "carla", saved_json_name + today.strftime("_%b_%d") + ".json"), "w") as f:
    json.dump(split, f, indent=2)

print("Split file saved to " + os.path.join(repo_dir, "maploc", "data", "carla", saved_json_name + today.strftime("_%b_%d") + ".json"))
