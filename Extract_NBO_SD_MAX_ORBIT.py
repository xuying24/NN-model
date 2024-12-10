#!/usr/bin/env python
import numpy as np
import os,linecache,csv,math,re



def get_all_file(base=None):
    result_paths = []
    if base is None:
        base = os.getcwd()
    for file in os.listdir(base):
        path = os.path.join(base, file)
        if os.path.isdir(path):
            result_paths.extend(get_all_file(path))
        else:
            if ".out" in file:
                result_paths.append(path)
    result_paths.sort(key=lambda x: os.path.split(x)[1])
    return result_paths

def file_processor(path):
    print(path)
    global line_Beta_begin
    with open(path, 'r')  as f:
        txt = f.readlines()

    #得到多重度
    dir_name, file_name = os.path.split(path)
    file_name = file_name.split('.')[0]
    Multi = int(file_name.split('-')[2].split('M')[1])

    block = 'Atom No    Charge        Core      Valence    Rydberg      Total'

    #得到原子总数
    i = 0
    for line in txt:
        i += 1
        if line.find(block)>-1:
            block_line = i

        if line.find(' * Total * ')>-1 or line.find('* Total * -0.99999')>-1 or line.find('* Total * -1.00001')>-1:
            total_line = i
            break
    nat = total_line - block_line - 3
    #print(nat)

    # 得到各元素原子数目
    line_num = 0
    atom_type_head = 0
    atom_type_1_num = 0
    atom_type_2_num = 0
    atom_type_3_num = 0
    atom_type_1 = ''
    atom_type_2 = ''
    atom_type_3 = ''
    element_num_dict={}

    for line in txt:
        line_num += 1

        if line.find(block) > -1 and (linecache.getline(path, line_num+ 3 + nat).find('* Total * -1.00000')>-1 or linecache.getline(path, line_num+ 3 + nat).find('* Total * -0.99999')>-1 or linecache.getline(path, line_num+ 3 + nat).find('* Total * -1.00001')>-1):
            #print("line_num:",line_num)
            atom_type_head = line_num + 2
            atom_type_tail = atom_type_head + nat
            atom_type_1 = linecache.getline(path, atom_type_head).split()[0]  # 第一种元素符号
            #print(atom_type_1)
            print('atom_type_1:',atom_type_1)
            for line_line in range(atom_type_head, atom_type_tail):
                atom_type = linecache.getline(path, line_line).split()[0]
                if atom_type == atom_type_1:
                    atom_type_1_num += 1  # 第一种元素原子数目

            #print('atom_type_1_num:',atom_type_1_num)
            atom_type_2 = linecache.getline(path, atom_type_head+atom_type_1_num).split()[0]  # 第二种元素符号
            for line_line in range(atom_type_head+atom_type_1_num, atom_type_tail):
                atom_type = linecache.getline(path, line_line).split()[0]
                if atom_type == atom_type_2:
                    atom_type_2_num += 1  # 第二种元素原子数目
                else:
                    atom_type_3=atom_type
                    atom_type_3_num += 1  # 第三种元素原子数目
            break

    print('atom_type_1_num:', atom_type_1_num)
    print('atom_type_2_num:', atom_type_2_num)
    print('atom_type_3_num:', atom_type_3_num)
    element_num_dict[atom_type_1]=atom_type_1_num
    element_num_dict[atom_type_2] = atom_type_2_num
    element_num_dict[atom_type_3] = atom_type_3_num
    print(element_num_dict)


    Rh_x = 0   #Rh原子数目
    V_y = 0     #V原子数目
    O_m = 0   #O原子数目
    # Rh_x = element_num_dict['Rh']
    V_y = element_num_dict['V']
    O_m = element_num_dict['O']





    print('Rh_x:',Rh_x)
    print('V_y:',V_y)
    print('O_m:',O_m)


   #确定包括所有轨道的Alfa 和 Beta 开端位置
    line_Alfa_Beta_begin = 0
    line_Alfa_begin = 0
    line_Beta_begin = 0
    find_line_Alfa_begin = 0
    for line in txt:
       line_Alfa_Beta_begin += 1
       if line.find('MOs in the NAO basis:')>-1 and find_line_Alfa_begin == 0:
           line_Alfa_begin = line_Alfa_Beta_begin
           find_line_Alfa_begin = 1
           #print(line_Alfa_begin)

       if line.find('MOs in the NAO basis:') > -1 and find_line_Alfa_begin == 1:
           line_Beta_begin = line_Alfa_Beta_begin
           #print(line_Beta_begin)

    #计算各轨道的序列号
    print(Multi)
    Alfa_HOMO_serial = int((17*Rh_x + 23*V_y +  2 - Multi) / 2 + Multi - 1)
    print('Alfa_HOMO_serial:',Alfa_HOMO_serial)
    Alfa_LUMO_serial = int((17*Rh_x + 23*V_y +  2 - Multi) / 2 + Multi)

    Beta_HOMO_serial = int((17*Rh_x + 23*V_y +  2 - Multi)/2)
    Beta_LUMO_serial = int((17*Rh_x + 23*V_y +  2 - Multi)/2 + 1)

    Orbit_total_num = Rh_x * 36 + V_y * 33 + O_m * 19

    Orbit_metal_num= Rh_x * 36 + V_y * 33

    # 得到金属原子及其编号的字符种类
    metal_type_num = []
    line_Alfa_begin_1st_block_head = line_Alfa_begin + 4
    line_Alfa_begin_1st_block_tail   = line_Alfa_begin_1st_block_head + Orbit_metal_num -1
    # print('line_Alfa_begin_1st_block_head:',line_Alfa_begin_1st_block_head)
    # print('line_Alfa_begin_1st_block_tail:',line_Alfa_begin_1st_block_tail)
    for line_line in range(line_Alfa_begin_1st_block_head, line_Alfa_begin_1st_block_tail):
        metal_type_num = metal_type_num + [linecache.getline(path, line_line).split("(")[0].split(".")[1]]
    metal_type_num = list(set(metal_type_num))
    #print('metal_type_num:',metal_type_num)

        #定位
    line_serial = 0
    Alfa_HOMO_block_seq_num = math.floor(Alfa_HOMO_serial/8 - 0.01)
    Alfa_LUMO_block_seq_num = math.floor(Alfa_LUMO_serial/8 - 0.01)

    Beta_HOMO_block_seq_num = math.floor(Beta_HOMO_serial / 8 - 0.01)
    Beta_LUMO_block_seq_num = math.floor(Beta_LUMO_serial / 8 - 0.01)

    for line in txt:
        line_serial += 1
        #Alfa HOMO  [Rh_5s、Rh_4d、V_4s、 V_3d]
        if line.find('NAO')>-1 and line.find('{}'.format(Alfa_HOMO_serial))>-1 and (line_Alfa_begin + 2 * (Alfa_HOMO_block_seq_num + 1)  + (Orbit_total_num + 1) * Alfa_HOMO_block_seq_num) == line_serial:
            Alfa_HOMO_serial_head = line_serial + 2
            Alfa_HOMO_serial_tail   =  Alfa_HOMO_serial_head + Orbit_total_num
            #print('Alfa_HOMO_serial_head',Alfa_HOMO_serial_head)
            dict_Alfa_HOMO_Rh_5s = {}
            dict_Alfa_HOMO_Rh_4d = {}
            dict_Alfa_HOMO_V_4s = {}
            dict_Alfa_HOMO_V_3d = {}
            dict_Alfa_HOMO_type_num = {}
            for type_num in metal_type_num:
                dict_Alfa_HOMO_Rh_5s['{}'.format(type_num)] = []
                dict_Alfa_HOMO_Rh_4d['{}'.format(type_num)] = []
                dict_Alfa_HOMO_V_4s['{}'.format(type_num)] = []
                dict_Alfa_HOMO_V_3d['{}'.format(type_num)] = []

            for line_line in range(Alfa_HOMO_serial_head,Alfa_HOMO_serial_tail):
                for type_num in metal_type_num:
                    if type_num.find('Rh')>-1:
                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path, line_line).find('5s') > -1:
                            block_seq_index = Alfa_HOMO_serial - Alfa_HOMO_block_seq_num * 8 - 1
                            dict_Alfa_HOMO_Rh_5s['{}'.format(type_num)] = dict_Alfa_HOMO_Rh_5s['{}'.format(type_num)]  + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path,line_line).find('4d') > -1:
                            block_seq_index = Alfa_HOMO_serial - Alfa_HOMO_block_seq_num * 8 - 1  # 在block中的列数-1
                            dict_Alfa_HOMO_Rh_4d['{}'.format(type_num)] = dict_Alfa_HOMO_Rh_4d['{}'.format(type_num)] + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

                    if type_num.find('V')>-1:
                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path, line_line).find('4s') > -1:
                            block_seq_index = Alfa_HOMO_serial - Alfa_HOMO_block_seq_num * 8 - 1

                            dict_Alfa_HOMO_V_4s['{}'.format(type_num)] = dict_Alfa_HOMO_V_4s['{}'.format(type_num)]  + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path,line_line).find('3d') > -1:
                            block_seq_index = Alfa_HOMO_serial - Alfa_HOMO_block_seq_num * 8 - 1  # 在block中的列数-1
                            dict_Alfa_HOMO_V_3d['{}'.format(type_num)] = dict_Alfa_HOMO_V_3d['{}'.format(type_num)] + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

            # print('metal_type_num:',metal_type_num)
            for type_num in metal_type_num:
                if type_num.find('Rh') > -1:
                    dict_Alfa_HOMO_type_num['{}'.format(type_num)] = sum(dict_Alfa_HOMO_Rh_5s['{}'.format(type_num)]) + sum(dict_Alfa_HOMO_Rh_4d['{}'.format(type_num)])
                if type_num.find('V') > -1:
                    dict_Alfa_HOMO_type_num['{}'.format(type_num)] = sum(dict_Alfa_HOMO_V_4s['{}'.format(type_num)]) + sum(dict_Alfa_HOMO_V_3d['{}'.format(type_num)])

            # print('dict_Alfa_HOMO_type_num:',dict_Alfa_HOMO_type_num)
            Max_contri_val_type_num_for_Alfa_HOMO = max(dict_Alfa_HOMO_type_num, key=lambda x: dict_Alfa_HOMO_type_num[x])
            Max_contri_val_for_Alfa_HOMO = dict_Alfa_HOMO_type_num[Max_contri_val_type_num_for_Alfa_HOMO] * 100

        # Alfa LUMO  [Rh_5s、Rh_4d、V_4s、 V_3d]
        if line.find('NAO')>-1 and line.find('{}'.format(Alfa_LUMO_serial))>-1 and (line_Alfa_begin + 2 * (Alfa_LUMO_block_seq_num + 1)  + (Orbit_total_num + 1) * Alfa_LUMO_block_seq_num) == line_serial:
            Alfa_LUMO_serial_head = line_serial + 2
            Alfa_LUMO_serial_tail   =  Alfa_LUMO_serial_head + Orbit_total_num
            #print('Alfa_HOMO_serial_head',Alfa_HOMO_serial_head)
            dict_Alfa_LUMO_Rh_5s = {}
            dict_Alfa_LUMO_Rh_4d = {}
            dict_Alfa_LUMO_V_4s = {}
            dict_Alfa_LUMO_V_3d = {}
            dict_Alfa_LUMO_type_num = {}
            for type_num in metal_type_num:
                dict_Alfa_LUMO_Rh_5s['{}'.format(type_num)] = []
                dict_Alfa_LUMO_Rh_4d['{}'.format(type_num)] = []
                dict_Alfa_LUMO_V_4s['{}'.format(type_num)] = []
                dict_Alfa_LUMO_V_3d['{}'.format(type_num)] = []

            for line_line in range(Alfa_LUMO_serial_head,Alfa_LUMO_serial_tail):
                for type_num in metal_type_num:
                    if type_num.find('Rh')>-1:
                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path, line_line).find('5s') > -1:
                            block_seq_index = Alfa_LUMO_serial - Alfa_LUMO_block_seq_num * 8 - 1
                            dict_Alfa_LUMO_Rh_5s['{}'.format(type_num)] = dict_Alfa_LUMO_Rh_5s['{}'.format(type_num)]  + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path,line_line).find('4d') > -1:
                            block_seq_index = Alfa_LUMO_serial - Alfa_LUMO_block_seq_num * 8 - 1  # 在block中的列数-1
                            dict_Alfa_LUMO_Rh_4d['{}'.format(type_num)] = dict_Alfa_LUMO_Rh_4d['{}'.format(type_num)] + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

                    if type_num.find('V')>-1:
                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path, line_line).find('4s') > -1:
                            block_seq_index = Alfa_LUMO_serial - Alfa_LUMO_block_seq_num * 8 - 1
                            dict_Alfa_LUMO_V_4s['{}'.format(type_num)] = dict_Alfa_LUMO_V_4s['{}'.format(type_num)]  + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path,line_line).find('3d') > -1:
                            block_seq_index = Alfa_LUMO_serial - Alfa_LUMO_block_seq_num * 8 - 1  # 在block中的列数-1
                            dict_Alfa_LUMO_V_3d['{}'.format(type_num)] = dict_Alfa_LUMO_V_3d['{}'.format(type_num)] + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

            for type_num in metal_type_num:
                if type_num.find('Rh') > -1:
                    dict_Alfa_LUMO_type_num['{}'.format(type_num)] = sum(dict_Alfa_LUMO_Rh_5s['{}'.format(type_num)]) + sum(dict_Alfa_LUMO_Rh_4d[type_num])
                if type_num.find('V') > -1:
                    dict_Alfa_LUMO_type_num['{}'.format(type_num)] = sum(dict_Alfa_LUMO_V_4s['{}'.format(type_num)]) + sum(dict_Alfa_LUMO_V_3d[type_num])

            Max_contri_val_type_num_for_Alfa_LUMO = max(dict_Alfa_LUMO_type_num, key=lambda x: dict_Alfa_LUMO_type_num[x])
            Max_contri_val_for_Alfa_LUMO = dict_Alfa_LUMO_type_num[Max_contri_val_type_num_for_Alfa_LUMO] * 100

        #Beta HOMO  [Rh_5s、Rh_4d、V_4s、 V_3d]
        if line.find('NAO')>-1 and line.find('{}'.format(Beta_HOMO_serial))>-1 and (line_Beta_begin + 2 * (Beta_HOMO_block_seq_num + 1)  + (Orbit_total_num + 1) * Beta_HOMO_block_seq_num) == line_serial:
            Beta_HOMO_serial_head = line_serial + 2
            Beta_HOMO_serial_tail   =  Beta_HOMO_serial_head + Orbit_total_num
            #print('Alfa_HOMO_serial_head',Alfa_HOMO_serial_head)
            dict_Beta_HOMO_Rh_5s = {}
            dict_Beta_HOMO_Rh_4d = {}
            dict_Beta_HOMO_V_4s = {}
            dict_Beta_HOMO_V_3d = {}
            dict_Beta_HOMO_type_num = {}
            for type_num in metal_type_num:
                dict_Beta_HOMO_Rh_5s['{}'.format(type_num)] = []
                dict_Beta_HOMO_Rh_4d['{}'.format(type_num)] = []
                dict_Beta_HOMO_V_4s['{}'.format(type_num)] = []
                dict_Beta_HOMO_V_3d['{}'.format(type_num)] = []

            for line_line in range(Beta_HOMO_serial_head,Beta_HOMO_serial_tail):
                for type_num in metal_type_num:
                    if type_num.find('Rh')>-1:
                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path, line_line).find('5s') > -1:
                            block_seq_index = Beta_HOMO_serial - Beta_HOMO_block_seq_num * 8 - 1
                            dict_Beta_HOMO_Rh_5s['{}'.format(type_num)] = dict_Beta_HOMO_Rh_5s['{}'.format(type_num)]  + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path,line_line).find('4d') > -1:
                            block_seq_index = Beta_HOMO_serial - Beta_HOMO_block_seq_num * 8 - 1  # 在block中的列数-1
                            dict_Beta_HOMO_Rh_4d['{}'.format(type_num)] = dict_Beta_HOMO_Rh_4d['{}'.format(type_num)] + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

                    if type_num.find('V')>-1:
                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path, line_line).find('4s') > -1:
                            block_seq_index = Beta_HOMO_serial - Beta_HOMO_block_seq_num * 8 - 1
                            dict_Beta_HOMO_V_4s['{}'.format(type_num)] = dict_Beta_HOMO_V_4s['{}'.format(type_num)]  + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path,line_line).find('3d') > -1:
                            block_seq_index = Beta_HOMO_serial - Beta_HOMO_block_seq_num * 8 - 1  # 在block中的列数-1
                            dict_Beta_HOMO_V_3d['{}'.format(type_num)] = dict_Beta_HOMO_V_3d['{}'.format(type_num)] + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

            for type_num in metal_type_num:
                if type_num.find('Rh') > -1:
                    dict_Beta_HOMO_type_num['{}'.format(type_num)] = sum(dict_Beta_HOMO_Rh_5s['{}'.format(type_num)]) + sum(dict_Beta_HOMO_Rh_4d[type_num])
                if type_num.find('V') > -1:
                    dict_Beta_HOMO_type_num['{}'.format(type_num)] = sum(dict_Beta_HOMO_V_4s['{}'.format(type_num)]) + sum(dict_Beta_HOMO_V_3d[type_num])

            Max_contri_val_type_num_for_Beta_HOMO = max(dict_Beta_HOMO_type_num, key=lambda x: dict_Beta_HOMO_type_num[x])
            Max_contri_val_for_Beta_HOMO = dict_Beta_HOMO_type_num[Max_contri_val_type_num_for_Beta_HOMO] * 100

        # Beta LUMO  [Rh_5s、Rh_4d、V_4s、 V_3d]
        if line.find('NAO')>-1 and line.find('{}'.format(Beta_LUMO_serial))>-1 and (line_Beta_begin + 2 * (Beta_LUMO_block_seq_num + 1)  + (Orbit_total_num + 1) * Beta_LUMO_block_seq_num) == line_serial:
            Beta_LUMO_serial_head = line_serial + 2
            Beta_LUMO_serial_tail   =  Beta_LUMO_serial_head + Orbit_total_num
            #print('Alfa_HOMO_serial_head',Alfa_HOMO_serial_head)
            dict_Beta_LUMO_Rh_5s = {}
            dict_Beta_LUMO_Rh_4d = {}
            dict_Beta_LUMO_V_4s = {}
            dict_Beta_LUMO_V_3d = {}
            dict_Beta_LUMO_type_num = {}
            for type_num in metal_type_num:
                dict_Beta_LUMO_Rh_5s['{}'.format(type_num)] = []
                dict_Beta_LUMO_Rh_4d['{}'.format(type_num)] = []
                dict_Beta_LUMO_V_4s['{}'.format(type_num)] = []
                dict_Beta_LUMO_V_3d['{}'.format(type_num)] = []

            for line_line in range(Beta_LUMO_serial_head,Beta_LUMO_serial_tail):
                for type_num in metal_type_num:
                    if type_num.find('Rh')>-1:
                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path, line_line).find('5s') > -1:
                            block_seq_index = Beta_LUMO_serial - Beta_LUMO_block_seq_num * 8 - 1
                            dict_Beta_LUMO_Rh_5s['{}'.format(type_num)] = dict_Beta_LUMO_Rh_5s['{}'.format(type_num)]  + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path,line_line).find('4d') > -1:
                            block_seq_index = Beta_LUMO_serial - Beta_LUMO_block_seq_num * 8 - 1  # 在block中的列数-1
                            dict_Beta_LUMO_Rh_4d['{}'.format(type_num)] = dict_Beta_LUMO_Rh_4d['{}'.format(type_num)] + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

                    if type_num.find('V')>-1:
                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path, line_line).find('4s') > -1:
                            block_seq_index = Beta_LUMO_serial - Beta_LUMO_block_seq_num * 8 - 1
                            dict_Beta_LUMO_V_4s['{}'.format(type_num)] = dict_Beta_LUMO_V_4s['{}'.format(type_num)]  + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

                        if linecache.getline(path, line_line).find('{}'.format(type_num)) > -1 and linecache.getline(path,line_line).find('3d') > -1:
                            block_seq_index = Beta_LUMO_serial - Beta_LUMO_block_seq_num * 8 - 1  # 在block中的列数-1
                            dict_Beta_LUMO_V_3d['{}'.format(type_num)] = dict_Beta_LUMO_V_3d['{}'.format(type_num)] + [(float(linecache.getline(path, line_line).split(")")[1].split()[block_seq_index]))**2]

            for type_num in metal_type_num:
                if type_num.find('Rh') > -1:
                    dict_Beta_LUMO_type_num['{}'.format(type_num)] = sum(dict_Beta_LUMO_Rh_5s['{}'.format(type_num)]) + sum(dict_Beta_LUMO_Rh_4d[type_num])
                if type_num.find('V') > -1:
                    dict_Beta_LUMO_type_num['{}'.format(type_num)] = sum(dict_Beta_LUMO_V_4s['{}'.format(type_num)]) + sum(dict_Beta_LUMO_V_3d[type_num])

            Max_contri_val_type_num_for_Beta_LUMO = max(dict_Beta_LUMO_type_num, key=lambda x: dict_Beta_LUMO_type_num[x])
            Max_contri_val_for_Beta_LUMO = dict_Beta_LUMO_type_num[Max_contri_val_type_num_for_Beta_LUMO] * 100

    return  Max_contri_val_type_num_for_Alfa_HOMO,Max_contri_val_for_Alfa_HOMO,Max_contri_val_type_num_for_Alfa_LUMO,Max_contri_val_for_Alfa_LUMO,Max_contri_val_type_num_for_Beta_HOMO,Max_contri_val_for_Beta_HOMO,Max_contri_val_type_num_for_Beta_LUMO,Max_contri_val_for_Beta_LUMO


def generate_result():
    paths = get_all_file()

    header = ["Cluster-name", "Max_contri_val_type_num_for_Alfa_HOMO", "Max_contri_val_for_Alfa_HOMO(%)", "Max_contri_val_type_num_for_Alfa_LUMO", "Max_contri_val_for_Alfa_LUMO(%)", "Max_contri_val_type_num_for_Beta_HOMO", "Max_contri_val_for_Beta_HOMO(%)", "Max_contri_val_type_num_for_Beta_LUMO", "Max_contri_val_for_Beta_LUMO(%)"]
    csv_output = open("NBO_Alfa_Beta_HOMO_LUMO_Rh_V_max_contri.csv", "w")
    csv_output.close()
    csv_output = open("NBO_Alfa_Beta_HOMO_LUMO_Rh_V_max_contri.csv", "a+", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_output, dialect='excel')
    csv_writer.writerow(header)

    for path in paths:
        row = []
        dir_name, file_name = os.path.split(path)
        file_name = file_name.split('.')[0]
        cluster = file_name.split('-')[0]
        multi = file_name.split('-')[2]
        num = file_name.split('-')[3]
        cluster_name = cluster + '-' + multi + '-' + num

        result = file_processor(path)
        row.append(cluster_name)
        row.append(result[0])
        row.append(result[1])
        row.append(result[2])
        row.append(result[3])
        row.append(result[4])
        row.append(result[5])
        row.append(result[6])
        row.append(result[7])
        csv_writer.writerow(row)
    csv_output.close()


if __name__ == "__main__":
    generate_result()
