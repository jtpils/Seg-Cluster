import os
import numpy as np
from plyfile import PlyData, PlyElement

ID_COLOR = {
 0 : [0,   0,   0  ],
 1 : [174, 199, 232],
 2 : [152, 223, 138],
 3 : [31,  119, 180],
 4 : [255, 187, 120],
 5 : [188, 189, 34 ],
 6 : [140, 86,  75 ],
 7 : [255, 152, 150],
 8 : [214, 39,  40 ],
 9 : [197, 176, 213],
 10: [148, 103, 189],
 11: [196, 156, 148],
 12: [23,  190, 207],
 13: [178, 76,  76 ],
 14: [247, 182, 210],
 15: [66,  188, 102],
 16: [219, 219, 141],
 17: [140, 57,  197],
 18: [202, 185, 52 ],
 19: [51,  176, 203],
 20: [200, 54,  131],
 21: [92,  193, 61 ],
 22: [78,  71,  183],
 23: [172, 114, 82 ],
 24: [255, 127, 14 ],
 25: [91,  163, 138],
 26: [153, 98,  156],
 27: [140, 153, 101],
 28: [158, 218, 229],
 29: [100, 125, 154],
 30: [178, 127, 135],
 31: [120, 185, 128],
 32: [146, 111, 194],
 33: [44,  160, 44 ],
 34: [112, 128, 144],
 35: [96,  207, 209],
 36: [227, 119, 194],
 37: [213, 92,  176],
 38: [94,  106, 211],
 39: [82,  84,  163],
 40: [100, 85,  144]}


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

def pc2ply(verts, output_file):
    """
    verts: numpy array (n, 4), last one is instance/label id
    output_file: string
    """
    with open(output_file, 'w') as f:
        f.write("ply\n");
        f.write("format ascii 1.0\n");
        f.write("element vertex {:d}\n".format(len(verts)));
        f.write("property float x\n");
        f.write("property float y\n");
        f.write("property float z\n");
        f.write("property uchar red\n");
        f.write("property uchar green\n");
        f.write("property uchar blue\n");
        f.write("end_header\n");
        for v in verts:
            r, g, b = ID_COLOR[int(v[4]%41)]
            f.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(v[0], v[1], v[2], r, g, b))

res_folder = 'output_nn'
ply_folder = 'output/scannet_ply'
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
        label = instances[instance_key]['label']
        output_file = os.path.join(output_dir, folder.split('.')[0], str(instance_key) + '.ply')
        pc2ply(np.concatenate([verts, np.ones_like(verts)*label], 1), output_file)



