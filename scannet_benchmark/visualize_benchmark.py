import os
import numpy as np
from plyfile import PlyData, PlyElement
from util_vis import pc2ply

class Benchmark_reader(object):
    def __init__(self, res_path):
        self.res_path = res_path
        self.res_txt = os.listdir(res_path)
    def __getitem__(self, txt_file):
        if txt_file  not in self.res_txt:
            print('{} not exist'.format(txt_file))
            return 0
        else:
            ret_instances = {}
            instances = open(os.path.join(self.res_path, txt_file)).readlines()
            for idx, instance in enumerate(instances):
                label = int(instance.split()[1])
                point_index = np.loadtxt(os.path.join(self.res_path, instance.split()[0]))
                ret_instances[idx] = {}
                ret_instances[idx]['points'] = point_index
                ret_instances[idx]['label'] = label
            return ret_instances

res_folder = 'res_benchmark'
ply_folder = 'data/scannet_ply'
output_dir = 'vis_benchmark'

reader_ins = Benchmark_reader(res_folder)
for folder in os.listdir(res_folder):
    print(folder)
    os.makedirs(os.path.join(output_dir, folder.split('.')[0]), exist_ok=True)

    # ply reader
    ply_file = os.path.join(ply_folder, folder.split('.')[0], folder.split('.')[0]+'.ply')
    ply_data = PlyData.read(ply_file)
    points = []
    for point in ply_data.elements[0].data:
        points.append([point[0], point[1], point[2]])
    points = np.array(points)

    # instance reader
    instances = reader_ins[folder]
    for instance_key in instances.keys():
        verts = points[instances[instance_key]['points'].nonzero()[0].astype(np.int32)]
        output_file = os.path.join(output_dir, folder.split('.')[0], str(instance_key) + '.ply')
        pc2ply(verts, color, output_file)



