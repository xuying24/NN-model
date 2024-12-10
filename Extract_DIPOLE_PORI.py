#!/usr/bin/env python
import numpy as np
import os,linecache,csv

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
    with open(path, 'r')  as f:
        txt = f.readlines()

    block_1 = 'Isotropic polarizability for W='
    block_2 = 'Dipole moment (field-independent basis, Debye):'

    #得到原子总数
    for line in txt:
        if line.find('NAtoms=')>-1:
            NAtoms = float(line.split()[1])
            break

    line_num = 0
    for line in txt:
        line_num += 1

        if line.find(block_1) > -1:
            #print(polar)
            polar = []
            polar = line.split()[5]
            #polar = polar[0]

        elif  line.find(block_2)>-1:
            Dipo = []
            #print(Dipo)
            dipo_line = line_num + 1
            Dipo = linecache.getline(path, dipo_line).split()[-1]


    return  polar,Dipo

def generate_result():
    paths = get_all_file()

    header = ["Cluster-name","Polarizability", "Dipole Moment"]
    csv_output = open("Polarizability_and_Dipole_Moment.csv", "w")
    csv_output.close()
    csv_output = open("Polarizability_and_Dipole_Moment.csv", "a+", newline="", encoding="utf-8")
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
        csv_writer.writerow(row)
    csv_output.close()


if __name__ == "__main__":
    generate_result()
