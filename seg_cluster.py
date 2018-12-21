import numpy as np
from scipy.spatial import distance
import os
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math

NUM_CATEGORY = 21
THRESH= 0.1
#NUM_CATEGORY = 26

def read_ply(ply_file):
    with open(ply_file, 'rb') as read_file:
        ply_data = PlyData.read(read_file)

    points = []
    for x,y,z,l,_,_ in ply_data['vertex']:
        points.append([x,y,z,l])
    points = np.array(points)
    return points

def bfs_cuda(points, old_coming, label, mask_seen):
    #type to float32
    # data is col-major order
    points = points.astype(np.float32)
    old_coming = np.array(old_coming).astype(np.float32)
    mask_seen = mask_seen.astype(np.float32)

    # mem alloc for gpu
    points_gpu = cuda.mem_alloc(points.nbytes)
    old_coming_gpu = cuda.mem_alloc(old_coming.nbytes)
    mask_seen_gpu = cuda.mem_alloc(mask_seen.nbytes)

    # copy to gpu
    cuda.memcpy_htod(points_gpu, points)
    cuda.memcpy_htod(old_coming_gpu, old_coming)
    cuda.memcpy_htod(mask_seen_gpu, mask_seen)

    # cuda program
    mod = SourceModule("""
            #include <stdio.h>
            #include <math.h>
            __global__ void bfs(int n, float* points, int old_coming_num, float *old_coming, float seed_label, float THRESH, float* mask_seen)
            {
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                if(idx >= n)
                    return ;
                float x     = points[idx*4 + 0];
                float y     = points[idx*4 + 1];
                float z     = points[idx*4 + 2];
                float label = points[idx*4 + 3];
                if(mask_seen[idx])
                    return ;
                for(int i=0; i < old_coming_num; i++)
                {
                    float x_old     = points[int(old_coming[i]) * 4 + 0];
                    float y_old     = points[int(old_coming[i]) * 4 + 1];
                    float z_old     = points[int(old_coming[i]) * 4 + 2];

                    if(seed_label == label)
                    {
                        float distance = sqrtf((x-x_old)*(x-x_old)+(y-y_old)*(y-y_old)+(z-z_old)*(z-z_old));
                        if (distance <= THRESH)
                        {
                            mask_seen[idx] = 1;
                            break;
                        }
                    }

                }
            }
            """)
    func = mod.get_function('bfs')
    num_grid = math.ceil(len(points) / 1024.0)
    func(np.int32(len(points)), points_gpu, np.int32(len(old_coming)), old_coming_gpu, np.float32(label), np.float32(THRESH),  mask_seen_gpu, block=(1024, 1, 1), grid=(num_grid, 1, 1))
    cuda.memcpy_dtoh(mask_seen, mask_seen_gpu)
    return mask_seen


def seg_cluster(points):
    instances = []
    mask_seen = np.zeros(points.shape[0])
    for seed_idx, seed in enumerate(points):
        print('{}/{}'.format(seed_idx, len(points)))
        if mask_seen[seed_idx] or seed[3] in [0,1]:
            mask_seen[seed_idx] = 1
            continue
        instances.append({'point':[seed_idx], 'label':seed[3]})

        old_coming = [seed_idx]
        mask_seen[seed_idx] = 1
        while True:
            mask_seen_new = bfs_cuda(points, old_coming, seed[3], mask_seen)
            new_coming = (mask_seen_new - mask_seen).nonzero()[0]
            mask_seen = mask_seen_new

            if len(new_coming) == 0:
                break
            else:
                instances[len(instances)-1]['point'].extend(new_coming)
                old_coming = new_coming


    return instances

def save2instance(output_dir, scene_name, instances, num_points):
    # write to file
    counter = 0
    f_scene = open(os.path.join(output_dir, scene_name + '.txt'), 'w')
    for i_sem in range(NUM_CATEGORY):
        for ins in instances:
            if i_sem == ins['label']:
                f_scene.write('{}_{:03d}.txt {} {}\n'.format(os.path.join(output_dir, 'predicted_masks', scene_name), counter, i_sem, 1.0))
                with open(os.path.join(output_dir, 'predicted_masks', '{}_{:03}.txt'.format(scene_name, counter)), 'w') as f:
                    for point_idx in range(num_points):
                        if point_idx in ins['point']:
                            f.write('1\n')
                        else:
                            f.write('0\n')
                counter += 1
    f_scene.close()

def visualize(points, instances):
    labels = np.zeros(len(points))
    for counter,ins in enumerate(instances):
        for idx in ins['point']:
            labels[int(idx)] = int(counter+1)
    import ipdb
    ipdb.set_trace()

ply_path = 'sem_output'
ply_files = os.listdir(ply_path)
for ply_file in ply_files:
    scene_name = ply_file.split('.')[0]
    points = read_ply(os.path.join(ply_path, ply_file))
    instances = seg_cluster(points)
    #visualize(points, instances)
    save2instance('output_sem', scene_name, instances, len(points))
