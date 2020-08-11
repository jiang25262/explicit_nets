import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from read_write_etc import *

gt_point_dir = "./evaluation/gt/"
bspnet_point_dir = "./im_test/"

eval_txt = open("./evaluation/all_vox256_img_test.txt","r")
eval_list = eval_txt.readlines()
eval_txt.close()
eval_list = [item.strip().split('/') for item in eval_list]
print(len(eval_list))

out_per_obj = open("result_per_obj_im.txt","w")
out_per_cat = open("result_per_category_im.txt","w")

category_list = ['02691156_airplane','02828884_bench','02933112_cabinet','02958343_car','03001627_chair','03211117_display','03636649_lamp','03691459_speaker','04090263_rifle','04256520_couch','04379243_table','04401088_phone','04530566_vessel']
category_name = [name[:8] for name in category_list]
category_num = [809, 364, 315, 1500, 1356, 219, 464, 324, 475, 635, 1702, 211, 388]
category_num_sum = [809, 1173, 1488, 2988, 4344, 4563, 5027, 5351, 5826, 6461, 8163, 8374, 8762]
category_chamfer_distance_sum = [0.0]*13
category_normal_consistency_sum = [0.0]*13
category_count =[0]*13

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

for idx in range(len(eval_list)):
    class_name = eval_list[idx][0]
    object_name = eval_list[idx][1]
    print(class_name,object_name)

    gt_pc_dir = gt_point_dir+class_name+'/'+object_name+'.ply'
    points_gt, normals_gt = read_ply_point_normal(gt_pc_dir)

    bspnet_pc_dir = bspnet_point_dir+class_name+'/'+object_name+'.ply'
    points_pd, normals_pd = read_ply_point_normal(bspnet_pc_dir)

    points_gt = torch.from_numpy(points_gt).to(device)
    normals_gt = torch.from_numpy(normals_gt).to(device)
    points_pd = torch.from_numpy(points_pd).to(device)
    normals_pd = torch.from_numpy(normals_pd).to(device)

    gt_pn = points_gt.size(0)
    pd_pn = points_pd.size(0)

    points_gt_mat = points_gt.view(gt_pn,1,3).repeat(1,pd_pn,1)
    points_pd_mat = points_pd.view(1,pd_pn,3).repeat(gt_pn,1,1)

    dist = torch.sum((points_gt_mat-points_pd_mat)**2,dim=2)

    match_pd_gt = torch.argmin(dist, dim=0)
    match_gt_pd = torch.argmin(dist, dim=1)

    dist_pd_gt = torch.mean((points_pd - points_gt[match_pd_gt])**2)*3
    dist_gt_pd = torch.mean((points_gt - points_pd[match_gt_pd])**2)*3

    normals_dot_pd_gt = torch.mean(torch.abs(torch.sum(normals_pd*normals_gt[match_pd_gt], dim=1)))
    normals_dot_gt_pd = torch.mean(torch.abs(torch.sum(normals_gt*normals_pd[match_gt_pd], dim=1)))

    chamfer_distance_out = (dist_pd_gt+dist_gt_pd).numpy()
    normal_consistency_out = ((normals_dot_pd_gt+normals_dot_gt_pd)/2).numpy()
    
    print(idx, chamfer_distance_out, normal_consistency_out)
    cat_id = category_name.index(class_name)
    category_count[cat_id] += 1
    category_chamfer_distance_sum[cat_id] += chamfer_distance_out
    category_normal_consistency_sum[cat_id] += normal_consistency_out
    out_per_obj.write( str(chamfer_distance_out)+' '+str(normal_consistency_out)+'\n' )

for i in range(13):
    out_per_cat.write(str(category_chamfer_distance_sum[i]/category_count[i]))
    out_per_cat.write('\t')
out_per_cat.write('\n')
for i in range(13):
    out_per_cat.write(str(category_normal_consistency_sum[i]/category_count[i]))
    out_per_cat.write('\t')
out_per_cat.write('\n')

out_per_obj.close()
out_per_cat.close()
