import os
import re
import csv


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
            if ".log" in file or ".out" in file:
                result_paths.append(path)
    result_paths.sort(key=lambda x: os.path.split(x)[1])
    return result_paths


def file_processor(path):
    print(path)
    block='Atom No    Charge        Core      Valence    Rydberg      Total      Density'
    Rh_st='Rh'
    V_st='V'
    O_st = 'O'
    sepst='--'
    file = open(path, "r")
    txt=file.read().splitlines()
    file.close()
    Cluster_charge = []
    Final_Cluster_charge = []
    Delta_charge=[]
    
    for line in txt :
      if line.find(block)>-1:
        Cluster_charge = []
        
      elif (line.find(Rh_st)> -1 and line[3:5] == 'Rh' and line[5:6] == ' ') or (line.find(V_st)> -1 and line[3:5] == ' V' and line[5:6] == ' ') :
        data=line[10:20]
#        print(data)
        Cluster_charge = Cluster_charge + [float(data)]
        
      elif line.find(' *******         Alpha spin orbitals         *******') > -1 :      
        break

    Final_Cluster_charge = Cluster_charge

    Cluster_max_charge = max(Final_Cluster_charge)

    Cluster_min_charge = min(Final_Cluster_charge)
  
    Delta_charge = Cluster_max_charge - Cluster_min_charge
    
    return 	Cluster_max_charge,Cluster_min_charge,Delta_charge


def generate_result():
    result_path = get_all_file()
    # result_output = open("result_output.txt", "w")
    csv_output = open("Get_cluster_Max_Min_charge.csv", "w")
    # result_output.close()
    csv_output.close()
    # result_output = open("result_output.txt", "a+")
    csv_output = open("Get_cluster_Max_Min_charge.csv", "a+", newline="", encoding="utf-8")
    header = ["Cluster_name",  "Cluster_max_charge", "Cluster_min_charge", "Delta_charge"]
    csv_writer = csv.writer(csv_output, dialect='excel')
    csv_writer.writerow(header)

    for path in result_path:
        row = []
        dir_name, file_name = os.path.split(path)
        file_name = file_name.split('.')[0]
        cluster = file_name.split('-')[0]
        multi   = file_name.split('-')[2]
        num    = file_name.split('-')[3]
        cluster_name = cluster + '-' + multi + '-' + num
        row.append(cluster_name)
        result = file_processor(path)
        
        Max_charge = result[0]
        Min_charge = result[1]
        row.append(Max_charge)
        row.append(Min_charge)
        Delta_charge = result[2]
        row.append(Delta_charge)
        csv_writer.writerow(row)
    csv_output.close()


if __name__ == "__main__":
    generate_result()
