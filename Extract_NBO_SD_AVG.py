#!/usr/bin/env python
import numpy as np
import os,linecache,csv

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
    with open(path, 'r')  as f:
        txt = f.readlines()

    block = 'Atom No    Charge        Core      Valence    Rydberg      Total'

    #得到原子总数
    i = 0
    for line in txt:
        i += 1
        if line.find(block)>-1:
            block_line = i
        if line.find('* Total * -1.00000')>-1 or line.find('* Total * -0.99999')>-1 or line.find('* Total * -1.00001')>-1:
            total_line = i
            break
    num_atoms = total_line - block_line - 3

    S = []
    D = []
    line_num = 0
    for line in txt:
        line_num += 1

        if  line.find(block)>-1 and (linecache.getline(path, line_num+ 3 + num_atoms).find('* Total * -1.00000')>-1 or linecache.getline(path, line_num+ 3 + num_atoms).find('* Total * -0.99999')>-1 or linecache.getline(path, line_num+ 3 + num_atoms).find('* Total * -1.00001')>-1):
            special_line = line_num + 7 + num_atoms
            if linecache.getline(path, special_line).find('Effective Core')>-1:
                sd_line_head = line_num + 16 + num_atoms
                sd_line_tail =  sd_line_head + num_atoms
            else:
                sd_line_head = line_num + 15 + num_atoms
                sd_line_tail = sd_line_head + num_atoms


            for line_line in range(sd_line_head, sd_line_tail):
                #print(float(linecache.getline(path, line_line).split('(')[1].split(')')[0]))
                if linecache.getline(path, line_line).find('Rh')>-1 or linecache.getline(path, line_line).find('V')>-1:
                   S = S + [float(linecache.getline(path, line_line).split('(')[1].split(')')[0])]
                   D = D + [float(linecache.getline(path, line_line).split('(')[2].split(')')[0])]
            break
    #print(S)
    S_avg = np.mean(S)
    D_avg = np.mean(D)

    return  S_avg,D_avg

def generate_result():
    paths = get_all_file()

    header = ["Cluster-name", "S_avg", "D_avg"]
    csv_output = open("NBO_s_and_d_avg.csv", "w")
    csv_output.close()
    csv_output = open("NBO_s_and_d_avg.csv", "a+", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_output, dialect='excel')
    csv_writer.writerow(header)

    for path in paths:
        row = []
        print(path)
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
