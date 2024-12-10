#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import sys
import linecache
from collections import defaultdict

''''
将用于计算VDE的log放置在该py脚本的相同目录下，直接运行即可，
注意log文件命名格式为如'Rh2V3--M3-tpsstpss-genecp-5.log'和'Rh2V3--M5-8-VDE-M4.log'
'''

def find_n_sub_str(s, sub, times, start=0):
    index = s.find(sub, start)
    if index != -1 and times > 0:
        return find_n_sub_str(s, sub, times - 1, index + 1)
    return index

def get_all_file(base=None):
    result_paths = []
    if base is None:
        base = os.getcwd()
    for file in os.listdir(base):
        path = os.path.join(base, file)
        if os.path.isdir(path):
            result_paths.extend(get_all_file(path))
        else:
            if ".log" in file:
                result_paths.append(path)
    result_paths.sort(key=lambda x: os.path.split(x)[1])
    return result_paths


def file_processor(path):

    if path.find('\\') > -1:
        file_name = path.split('\\')[-1]
    elif path.find('/') > -1:
        file_name = path.split('/')[-1]

    if file_name.find('VDE') == -1:
        num = file_name.split('-')[3].split('.')[0]
        multi = int(file_name.split('M')[1].split('-')[0])

    elif file_name.find('VDE') > -1:
        num = file_name.split('-')[3].split('.')[0]
        multi = int(file_name.split('M')[1].split('-')[0])



    block_2='NAtoms'
    block_1=' Mulliken charges:'
    block_3=' Mulliken charges and spin densities:'
    block_4='Sum of Mulliken charges'
    block_5=' Mulliken atomic charges:'
    block_6=' Sum of Mulliken atomic charges'
    block_7='Multiplicity'
    block_8=' Sum of electronic and zero-point Energies='
    sepst='--'
    file = open(path, "r")
    txt=file.read().splitlines()
    file.close()
    energy = []
    
    line_num=0
    atom_type_head = 0
    atom_type_1_num = 0
    atom_type_2_num = 0
    atom_type_3_num = 0
    atom_type_1 = ''
    atom_type_2 = ''
    atom_type_3 = ''
    element_num_dict={}

    for line in txt :
        line_num+=1
        if line.find(block_2)>-1:
            nat = int(line.split()[1])   #原子数目
            #print('nat:',nat)

        if line.find(block_7)>-1:
            charge = line.split()[2]      #电荷数

#         if line.find(block_3)>-1 or line.find(block_5)>-1 or line.find(block_1)>-1:
# #            print("line_num:",line_num)
#             atom_type_head = line_num + 2
#             atom_type_tail = atom_type_head + nat
#             atom_type_1 = linecache.getline(path, atom_type_head).split()[1]  #第一种元素符号
# #            print('atom_type_1:',atom_type_1)
#             for line_line in range(atom_type_head,atom_type_tail):
#                 atom_type = linecache.getline(path, line_line).split()[1]
#                 if atom_type == atom_type_1:
#                     atom_type_1_num += 1    #第一种元素原子数目
#
#             atom_type_2 = linecache.getline(path, atom_type_head + atom_type_1_num).split()[1]  # 第二种元素符号
#             for line_line in range(atom_type_head + atom_type_1_num, atom_type_tail):
#                 atom_type = linecache.getline(path, line_line).split()[1]
#                 #print(linecache.getline(path, line_line).split())
#                 if atom_type == atom_type_2:
#                     atom_type_2_num += 1  # 第二种元素原子数目
#                 else:
#                     atom_type_3 = atom_type
#                     atom_type_3_num += 1  # 第三种元素原子数目
#             break
#     print('atom_type_1:', atom_type_1)
#     print('atom_type_2:', atom_type_2)
#     print('atom_type_3:', atom_type_3)
#     print('atom_type_1_num:',atom_type_1_num)
#     print('atom_type_2_num:', atom_type_2_num)
#     print('atom_type_3_num:', atom_type_3_num)
#     element_num_dict[atom_type_1] = atom_type_1_num
#     element_num_dict[atom_type_2] = atom_type_2_num
#     element_num_dict[atom_type_3] = atom_type_3_num


    print(path)
    i = 0
    for line in txt :
        i = i + 1

    #提取零点矫正能
    for line in txt :
#        print(line)
        if line.find(block_8) > -1:
            energy.clear()
            energy = energy + [float(line.split('=')[1])]


    #得到团簇名称
    dir_name, file_name = os.path.split(path)
    file_name = file_name.split('.')[0]
    cluster_name = file_name.split('-')[0]
    # cluster_name=str(atom_type_1) + str(element_num_dict[atom_type_1]) + str(atom_type_2) + str(element_num_dict[atom_type_2])  + str(atom_type_3) + str(element_num_dict[atom_type_3])
    # print('cluster_name:',cluster_name)
    # if nat == atom_type_1_num and atom_type_1_num != 1:
    #     cluster_name = str(atom_type_1) + str(atom_type_1_num) + str('-M{}'.format(multi)) + str('-{}'.format(num))
    # elif nat == atom_type_1_num and atom_type_1_num == 1:
    #     cluster_name = str(atom_type_1) + str('-M{}'.format(multi)) + str('-{}'.format(num))
    # elif atom_type_1_num == 1 and atom_type_2_num == 1:
    #     cluster_name = str(atom_type_1) + str(atom_type_2) + str('-M{}'.format(multi)) + str('-{}'.format(num))
    # elif atom_type_1_num == 1 and atom_type_2_num != None and atom_type_2_num !=1:
    #     cluster_name = str(atom_type_1) + str(atom_type_2) + str(atom_type_2_num) + str('-M{}'.format(multi)) + str('-{}'.format(num))
    # elif atom_type_1_num != 1 and atom_type_2_num ==1:
    #     cluster_name = str(atom_type_1) + str(atom_type_1_num) + str(atom_type_2) + str('-M{}'.format(multi)) + str('-{}'.format(num))
    # elif atom_type_1_num !=1 and atom_type_2_num != None and atom_type_2_num !=1:
    #     cluster_name = str(atom_type_1)+str(atom_type_1_num)+str(atom_type_2)+str(atom_type_2_num) + str('-M{}'.format(multi)) + str('-{}'.format(num))

    return 	nat,atom_type_1_num,atom_type_2_num,atom_type_1,atom_type_2,multi,energy,charge,cluster_name


def generate_result():
    result_path = get_all_file()
#    csv_output = open("fil_structs.xyz.0", "a+", newline="", encoding="utf-8")

    dic_energy = {}       #定义存储团簇所有能量的字典
    log_path = {}       #定义存储路径的字典
    cluster_name = []     #定义存储团簇名称的列表

    #得到团簇名称列表和不同团簇的空字典
    for path in result_path:
        result = file_processor(path)
        #记录团簇名称
        cluster_name = cluster_name + [result[8]]
        #定义不同键的空字典
        dic_energy['{}'.format(result[8])] = []
        log_path['{}'.format(result[8])] = []

    for path in result_path:
        result = file_processor(path)
        if len(result[6])  == 0:
            print("The {} have no ' Sum of electronic and zero-point Energies='".format(path))
            sys.exit(0)
        # 相同团簇不同结构的能量归类
        dic_energy['{}'.format(result[8])].append(float(result[6][0]))
        log_path['{}'.format(result[8])].append(path)

    #去掉cluster_name中的重复元素
    all_cluster_name = list(set(cluster_name))
#    print(all_cluster_name)

    # 构建存储团簇的最低能量的字典
    dic_energy_min = {}  # 定义存储团簇最低能量的字典
    for cluster_name_v in all_cluster_name:
        energy_min = min(dic_energy[cluster_name_v])
        dic_energy_min['{}'.format(cluster_name_v)] = []
        dic_energy_min['{}'.format(cluster_name_v)].append(energy_min)

    with open("All_structs_VDE.txt", 'w') as f:
        f.truncate()

    # 对能量排序
    dic_energy_sorted = {}  # 定义存储同一团簇从小到大排序好的能量字典
    for cluster_name_v in all_cluster_name:
        dic_energy_np = np.array(dic_energy[cluster_name_v])
        dic_energy_sorted['{}'.format(cluster_name_v)] = sorted(dic_energy_np)

        #构建存储团簇用于计算VDE的能量的字典
        # print(cluster_name_v)
        # print(dic_energy_sorted['{}'.format(cluster_name_v)])
        # print(dic_energy_sorted['{}'.format(cluster_name_v)][1])
        VDE = (float(dic_energy_sorted['{}'.format(cluster_name_v)][1]) - float(dic_energy_min['{}'.format(cluster_name_v)][0])) * 27.2114
        print('{:<15}       VDE =       {:.3f}   eV'.format(cluster_name_v, VDE))
        with open("All_structs_VDE.txt", 'a+') as f:
            f.write('{:<15}         {:6}      {:.3f}       eV\n'.format(cluster_name_v, 'VDE =',VDE))

        # dic_VDE['{}'.format(cluster_name_v)] = []
        # dic_VDE['{}'.format(cluster_name_v)].append(VDE)



if __name__ == "__main__":
    generate_result()
