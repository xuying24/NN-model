#!/usr/bin python
# coding: utf-8
from collections import defaultdict
import os, re, csv, linecache, sys
import numpy as np
import pandas as pd

non_metallic_elements_in_clusters = ['O', 'S', 'C', 'N', 'H']

df = pd.read_excel('./Complete_rate.xlsx')
df_cluster_in_excel = df['cluster']
df_Molecule_in_excel = df['molecule']
df_lgk1_in_excel = df['lgk1']
df_Uplimit = df['Uplimit (F=1orT=0)']


Cn = {'CH4':1, 'CH3CH3':2, 'CH3CH2CH3':3, 'CH3CH2CH2CH3':4}
Molecule_polar = {'CH4':2.593, 'CH3CH3':4.47, 'CH3CH2CH3':6.29, 'CH3CH2CH2CH3':8.20}
Molecule_IP = {'CH4':12.61, 'CH3CH3':11.56, 'CH3CH2CH3':10.95, 'CH3CH2CH2CH3':10.53}
CH4_homo = -10.784  
CH4_lumo = 0.27 

C2H6_homo = -9.434  
C2H6_lumo = 0.267

C3H8_homo = -9.001  
C3H8_lumo = 0.226 

C4H10_homo = -8.86  
C4H10_lumo = 0.254  

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

def Get_HOMO_LUMO(path):
    block='Population analysis'
    eigen='eigenvalues'
    Alpha_occst='Alpha  occ. eigenvalues'
    Alpha_virst='Alpha virt. eigenvalues'
    Beta_occst ='Beta  occ. eigenvalues'
    Beta_virst ='Beta virt. eigenvalues'
    sepst='--'
    file = open(path, "r", encoding='utf-8')
    txt=file.read().splitlines()
    file.close()
    for line in txt :
      if line.find(block) > -1 or line.find(eigen) > -1 :
        if line.find(block) > -1 :
          Alpha_eocc=[]
          Alpha_evir=[]
          Beta_eocc=[]
          Beta_evir=[]
    ####################### Alpha ################################
        elif line.find(Alpha_occst) > -1 :
          data=line.split(sepst)[1].replace("-"," -")

          Alpha_eocc=Alpha_eocc+[float(i) for i in data.split()]
        elif line.find(Alpha_virst) > -1 :
          data=line.split(sepst)[1].replace("-"," -")
          Alpha_evir=Alpha_evir+[float(i) for i in data.split()]
    
    ######################## Beta ################################
        elif line.find(Beta_occst) > -1 :
          data=line.split(sepst)[1].replace("-"," -")

          Beta_eocc=Beta_eocc+[float(i) for i in data.split()]
        elif line.find(Beta_virst) > -1 :
          data=line.split(sepst)[1].replace("-"," -")
          Beta_evir=Beta_evir+[float(i) for i in data.split()]
    
    if len(Alpha_eocc) > 0 :  Alpha_HOMO =max(Alpha_eocc)
    if len(Alpha_evir) > 0 :  Alpha_LUMO =min(Alpha_evir)
    
    if len(Beta_eocc) > 0 :  Beta_HOMO =max(Beta_eocc)
    if len(Beta_eocc) == 0:   Beta_HOMO = Alpha_HOMO

    if len(Beta_evir) > 0 :  Beta_LUMO =min(Beta_evir)
    if len(Beta_evir) == 0:  Beta_LUMO = Alpha_LUMO
    
    return 	Alpha_HOMO, Alpha_LUMO, Beta_HOMO, Beta_LUMO

def Get_Max_Min_charge(path, cluster_name):
    atom_counts = defaultdict(int)
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', cluster_name)
    for element, count in elements:
        if count == '':
            count = 1
        else:
            count = int(count)
        atom_counts[element] += count
    atom_counts = dict(atom_counts)
    Natoms = sum(atom_counts.values())

    block = '  Atom No    Charge        Core      Valence    Rydberg      Total'
    file = open(path, "r")
    txt = file.read().splitlines()
    file.close()
    line_num = 0
    for line in txt:
        line_num += 1

        if line.find(block) > -1 and (round(float(linecache.getline(path, line_num+ 3 + Natoms).split()[3])) == -1 or round(float(linecache.getline(path, line_num+ 3 + Natoms).split()[3])) == 1):
            Metal_atoms_charge = []
            line_head = line_num + 2
            line_tail   = line_head + Natoms
            for line_line in range(line_head, line_tail):
                # print(linecache.getline(path, line_line))
                if linecache.getline(path, line_line).split()[0] not in non_metallic_elements_in_clusters:
                    Metal_atoms_charge = Metal_atoms_charge + [float(linecache.getline(path, line_line).split()[2])]

        elif line.find(' *******         Alpha spin orbitals         *******') > -1:
            break

    Metal_atoms_max_charge = max(Metal_atoms_charge)
    Metal_atoms_min_charge = min(Metal_atoms_charge)

    return Metal_atoms_max_charge, Metal_atoms_min_charge

def Get_polar_dipole_rota_factor(path):
    with open(path, 'r', encoding='utf-8')  as f:
        txt = f.readlines()

    block_1 = 'Isotropic polarizability for W='
    block_2 = 'Dipole moment (field-independent basis, Debye):'
    block_3 = ' Rotational constants (GHZ):'
    polar = 0
    Dipo = 0
    Rota_factor = 0

    #得到原子总数
    for line in txt:
        if line.find('NAtoms=')>-1:
            NAtoms = float(line.split()[1])
            break

    line_num = 0
    for line in txt:
        line_num += 1

        if line.find(block_1) > -1:
            polar = line.split()[5]

        elif  line.find(block_2)>-1:
            dipo_line = line_num + 1
            Dipo = linecache.getline(path, dipo_line).split()[-1]

        elif line.find(block_3)>-1:
            Rota_const_list = line.split(':')[1].split()
            if Rota_const_list[0] != '************':
                Rota_factor = (float(Rota_const_list[0]) - float(Rota_const_list[2])) / (float(Rota_const_list[0]) + float(Rota_const_list[1]) + float(Rota_const_list[2]))

    return polar, Dipo, Rota_factor

def Avg_NEC_s_d(path, cluster_name):
    with open(path, 'r')  as f:
        txt = f.readlines()
    block = 'Atom No    Charge        Core      Valence    Rydberg      Total'

    #得到原子总数
    atom_counts = defaultdict(int)
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', cluster_name)
    for element, count in elements:
        if count == '':
            count = 1
        else:
            count = int(count)
        atom_counts[element] += count
    atom_counts = dict(atom_counts)
    Natoms = sum(atom_counts.values())

    line_num = 0
    for line in txt:
        line_num += 1

        if  line.find(block)>-1 and (round(float(linecache.getline(path, line_num + 3 + Natoms).split()[3])) == -1 or round(float(linecache.getline(path, line_num + 3 + Natoms).split()[3])) == 1 ):
            S = []
            D = []
            special_line = line_num + 7 + Natoms
            if linecache.getline(path, special_line).find('Effective Core')>-1:
                sd_line_head = line_num + 16 + Natoms
                sd_line_tail =  sd_line_head + Natoms
            else:
                sd_line_head = line_num + 15 + Natoms
                sd_line_tail = sd_line_head + Natoms

            for line_line in range(sd_line_head, sd_line_tail):
                if linecache.getline(path, line_line).split()[0] not in non_metallic_elements_in_clusters:
                   S = S + [float(linecache.getline(path, line_line).split('(')[1].split(')')[0])]
                   D = D + [float(linecache.getline(path, line_line).split('(')[2].split(')')[0])]
            break

    S_avg = np.mean(S)
    D_avg = np.mean(D)

    return S_avg, D_avg

def Get_SCF_energy_for_VDE_or_VEA(path):

    block = ' SCF Done:  E('
    #  提取SCF Done 能量
    file = open(path, "r", encoding='utf-8')
    txt = file.read().splitlines()
    file.close()
    for line in txt:
        if line.find(block) > -1:
            energy = [float(line.split('=')[1].split()[0])]

    return energy

def generate_result(df_cluster_in_excel, df_Molecule_in_excel, df_lgk1_in_excel, df_Uplimit):
    result_path = get_all_file()
    csv_output = open("Features_table.csv", "w")
    csv_output.close()
    csv_output = open("Features_table.csv", "a+", newline="", encoding="utf-8")
    header = ["Reaction_serials", "Clusters_serials", "Clusters",  "Molecule",  "Molecule-polar", "Molecule_IP", "M-charge", "M-E(LUMO)", "M-E(HOMO)", "M-E(H-L gap)", "E(LC-HM)", "E(LM-HC)",
    "Q-max", "Q-min", "M-Polar", "Dipole", "Rota_factor", "Avg.NEC(s)", "Avg.NEC(d)", "VDE", "VEA", "lgk1", "Uplimit (F=1orT=0)"]
    csv_writer = csv.writer(csv_output, dialect='excel')
    csv_writer.writerow(header)

    dic_cluster_serials = defaultdict(float)
    dic_cluster_name = defaultdict(float)
    dic_molecule_homo = defaultdict(float)
    dic_molecule_lumo = defaultdict(float)
    dic_cluster_charge = defaultdict(float)
    dic_cluster_homo = defaultdict(float)
    dic_cluster_lumo = defaultdict(float)
    dic_cluster_homo_lumo_gap = defaultdict(float)
    dic_max_min_charge_metal_atoms = defaultdict(float)
    dic_polar_dipole_rota_factor = defaultdict(float)
    dic_avg_s_d = defaultdict(float)
    dic_VDE = defaultdict(float)
    dic_VEA = defaultdict(float)
    dic_energy_for_VDE = defaultdict(float)
    dic_energy_for_VDE_sorted = defaultdict(float)
    dic_energy_for_VEA = defaultdict(float)
    dic_energy_for_VEA_sorted = defaultdict(float)

    clusters_in_result_path = []
    All_cluster_name_in_files = []
    for path in result_path:
        dir_name, file_name = os.path.split(path)
        file_name_0 = file_name.split('.')[0]
        if path.find('.log') > -1 and path.find('VDE') == -1 and path.find('VEA') == -1 and (file_name_0.find('B')>-1 or file_name_0.find('A') > -1):
            clusters_in_result_path.append(path)
            cluster_name = file_name.split('_')[1]
            All_cluster_name_in_files.append(cluster_name)

            dic_cluster_serials[cluster_name] = []
            dic_cluster_name[cluster_name] = []
            dic_cluster_charge[cluster_name] = []
            dic_cluster_homo[cluster_name] = []
            dic_cluster_lumo[cluster_name] = []
            dic_cluster_homo_lumo_gap[cluster_name] = []
            dic_max_min_charge_metal_atoms[cluster_name] = []
            dic_polar_dipole_rota_factor[cluster_name] = []
            dic_avg_s_d[cluster_name] = []
            dic_VDE[cluster_name] = []
            dic_VEA[cluster_name] = []

            dic_energy_for_VDE[cluster_name] = []
            dic_energy_for_VEA[cluster_name] = []

        if path.find('.log') > -1 and path.find('VDE') == -1 and path.find('VEA') == -1 and (file_name_0.find('B') == -1 or file_name_0.find('A') == -1):
            molecule_name = file_name_0
            dic_molecule_homo[molecule_name] = []
            dic_molecule_lumo[molecule_name] = []

    file_num = 0
    last_cluster_name = ''
    last_cluster_name_Charge = 0
    for path in result_path:
        # For Molecules HOMO LUMO
        dir_name, file_name = os.path.split(path)
        file_name_0 = file_name.split('.')[0]
        if path.find('.log') > -1 and path.find('VDE') == -1 and path.find('VEA') == -1 and (file_name_0.find('B') == -1 or file_name_0.find('A') == -1):
            molecule_name = file_name_0
            Get_HOMO_LUMO_result = Get_HOMO_LUMO(path)
            Alpha_HOMO = float(Get_HOMO_LUMO_result[0]) * 27.2113682
            Alpha_LUMO = float(Get_HOMO_LUMO_result[1]) * 27.2113682
            Beta_HOMO = float(Get_HOMO_LUMO_result[2]) * 27.2113682
            Beta_LUMO = float(Get_HOMO_LUMO_result[3]) * 27.2113682
            if Alpha_LUMO > Beta_LUMO:
                Lumo = Beta_LUMO
                dic_molecule_lumo[molecule_name].append(Lumo)
            else:
                Lumo = Alpha_LUMO
                dic_molecule_lumo[molecule_name].append(Lumo)

            if Alpha_HOMO > Beta_HOMO:
                Homo = Alpha_HOMO
                dic_molecule_homo[molecule_name].append(Homo)
            else:
                Homo = Beta_HOMO
                dic_molecule_homo[molecule_name].append(Homo)

        # For clusters
        dir_name, file_name = os.path.split(path)
        file_name_0 = file_name.split('.')[0].split('_')[0]
        file_num += 1

        if path.find('.log') > -1 and path.find('VDE') == -1 and path.find('VEA') == -1 and (file_name_0.find('B')>-1 or file_name_0.find('A') > -1):
            # print(last_cluster_name)
            dir_name, file_name = os.path.split(path)
            file_name = file_name.split('.')[0]
            cluster_name = file_name.split('_')[1]
            print('cluster_name_files(log or out)', cluster_name)
            Serials = file_name.split('_')[0]

            with open(path, 'r', encoding='utf-8') as f:
                txt = f.readlines()
            for line in txt:
                if line.find(' Charge =') > -1:
                    Charge = float(line.split()[2])
                    print(path)
                    print(cluster_name)
                    print(Charge)
                    dic_cluster_charge[cluster_name] = Charge
                    # print(Charge)
                    break

            Final_dir_name, Final_file_name = os.path.split(clusters_in_result_path[-1])
            Final_file_name = Final_file_name.split('.')[0]
            Final_cluster_name = Final_file_name.split('_')[1]


            dic_cluster_serials[cluster_name].append(Serials)
            dic_cluster_name[cluster_name].append(cluster_name)

            # HOMO-LUMO gap Cluster
            Get_HOMO_LUMO_result = Get_HOMO_LUMO(path)
            Alpha_HOMO = float(Get_HOMO_LUMO_result[0]) * 27.2113682
            Alpha_LUMO = float(Get_HOMO_LUMO_result[1]) * 27.2113682
            Beta_HOMO = float(Get_HOMO_LUMO_result[2]) * 27.2113682
            Beta_LUMO = float(Get_HOMO_LUMO_result[3]) * 27.2113682
            if Alpha_LUMO>Beta_LUMO:
                Lumo = Beta_LUMO
                dic_cluster_lumo[cluster_name].append(Lumo)
            else:
                Lumo = Alpha_LUMO
                dic_cluster_lumo[cluster_name].append(Lumo)

            if Alpha_HOMO>Beta_HOMO:
                Homo = Alpha_HOMO
                dic_cluster_homo[cluster_name].append(Homo)
            else:
                Homo = Beta_HOMO
                dic_cluster_homo[cluster_name].append(Homo)

            H_L_gap = Lumo - Homo
            dic_cluster_homo_lumo_gap[cluster_name].append(H_L_gap)

            # Dipole, Polar, Rotational_factor
            Get_polar_dipole_rota_factor_result = Get_polar_dipole_rota_factor(path)
            dic_polar_dipole_rota_factor[cluster_name].append(Get_polar_dipole_rota_factor_result[0])
            dic_polar_dipole_rota_factor[cluster_name].append(Get_polar_dipole_rota_factor_result[1])
            dic_polar_dipole_rota_factor[cluster_name].append(Get_polar_dipole_rota_factor_result[2])

        # VDE of Clusters
        if path.find('.log') > -1 and path.find('VEA') == -1:
            Get_SCF_energy_for_VDE_or_VEA_result = Get_SCF_energy_for_VDE_or_VEA(path)
            dic_energy_for_VDE[cluster_name].append(Get_SCF_energy_for_VDE_or_VEA_result)

            if (last_cluster_name != cluster_name and file_num != 1) or (Final_cluster_name == cluster_name):
                dic_energy_for_VDE_np = np.array(dic_energy_for_VDE[last_cluster_name])
                # print(dic_energy_for_VDE_np)
                dic_energy_for_VDE_sorted[last_cluster_name] = sorted(dic_energy_for_VDE_np)   #能量从小到大排序

                if len(dic_energy_for_VDE_sorted[last_cluster_name]) < 2:
                    print("Check {} VDE file".format(last_cluster_name))
                VDE = (float(dic_energy_for_VDE_sorted[last_cluster_name][1]) - float(dic_energy_for_VDE_sorted[last_cluster_name][0])) * 27.2114
                if Final_cluster_name == cluster_name:
                    dic_VDE[last_cluster_name] = []
                    dic_VDE[last_cluster_name].append(VDE)
                else:
                    dic_VDE[last_cluster_name].append(VDE)
            # last_cluster_name = cluster_name

        # VEA of Clusters
        if path.find('.log') > -1 and path.find('VDE') == -1:
            Get_SCF_energy_for_VDE_or_VEA_result = Get_SCF_energy_for_VDE_or_VEA(path)
            dic_energy_for_VEA[cluster_name].append(Get_SCF_energy_for_VDE_or_VEA_result)

            if (last_cluster_name != cluster_name and file_num != 1 and last_cluster_name_Charge < 0) or (Final_cluster_name == cluster_name and last_cluster_name_Charge < 0):
                dic_energy_for_VEA_np = np.array(dic_energy_for_VEA[last_cluster_name])
                dic_energy_for_VEA_sorted[last_cluster_name] = sorted(dic_energy_for_VEA_np)  # 能量从小到大排序
                if len(dic_energy_for_VEA_sorted[last_cluster_name]) < 2:
                    print("Check {} VEA file".format(last_cluster_name))
                VEA = (dic_energy_for_VEA_sorted[last_cluster_name][1] - dic_energy_for_VEA_sorted[last_cluster_name][0]) * 27.2114
                if Final_cluster_name == cluster_name:
                    dic_VEA[last_cluster_name]=[]
                    dic_VEA[last_cluster_name].append(VEA)
                else:
                    dic_VEA[last_cluster_name].append(VEA)
            elif (last_cluster_name != cluster_name and file_num != 1 and last_cluster_name_Charge >= 0) or (Final_cluster_name == cluster_name and last_cluster_name_Charge >= 0):
                dic_energy_for_VEA_np = np.array(dic_energy_for_VEA[last_cluster_name])
                dic_energy_for_VEA_sorted[last_cluster_name] = sorted(dic_energy_for_VEA_np)  # 能量从小到大排序
                if len(dic_energy_for_VEA_sorted[last_cluster_name]) < 2:
                    print("Check {} VEA file".format(last_cluster_name))
                VEA = (min(dic_energy_for_VEA_sorted[last_cluster_name]) - max(dic_energy_for_VEA_sorted[last_cluster_name])) * 27.2114
                if Final_cluster_name == cluster_name:
                    dic_VEA[last_cluster_name]=[]
                    dic_VEA[last_cluster_name].append(VEA)
                else:
                    dic_VEA[last_cluster_name].append(VEA)

        last_cluster_name = cluster_name
        last_cluster_name_Charge = Charge

    for path in result_path:
        if path.find('NBO.out') > -1:
            dir_name, file_name = os.path.split(path)
            file_name = file_name.split('.')[0]
            cluster_name = file_name.split('_')[1]

            # Max_Min_charge  of metal atoms
            Get_Max_Min_charge_result = Get_Max_Min_charge(path, cluster_name)
            dic_max_min_charge_metal_atoms[cluster_name].append(Get_Max_Min_charge_result[0])
            dic_max_min_charge_metal_atoms[cluster_name].append(Get_Max_Min_charge_result[1])

            # Avg_NEC_s_d of metal atoms
            Avg_NEC_s_d_result = Avg_NEC_s_d(path, cluster_name)
            dic_avg_s_d[cluster_name].append(Avg_NEC_s_d_result[0])
            dic_avg_s_d[cluster_name].append(Avg_NEC_s_d_result[1])

    dic_molecule_homo = dict(dic_molecule_homo)
    dic_molecule_lumo = dict(dic_molecule_lumo)
    dic_cluster_charge = dict(dic_cluster_charge)
    print(dic_cluster_charge)
    dic_cluster_homo = dict(dic_cluster_homo)
    dic_cluster_lumo = dict(dic_cluster_lumo)
    dic_cluster_homo_lumo_gap = dict(dic_cluster_homo_lumo_gap)
    dic_polar_dipole_rota_factor = dict(dic_polar_dipole_rota_factor)
    dic_VDE = dict(dic_VDE)
    dic_VEA = dict(dic_VEA)
    dic_max_min_charge_metal_atoms = dict(dic_max_min_charge_metal_atoms)
    dic_avg_s_d = dict(dic_avg_s_d)

    no_log_and_out_files_of_cluster = []
    for i_Reaction_serials in range(0, len(df_cluster_in_excel)):
        cluster_name = df_cluster_in_excel[i_Reaction_serials]
        Molecule = df_Molecule_in_excel[i_Reaction_serials]
        lgk1 = df_lgk1_in_excel[i_Reaction_serials]
        Uplimit = df_Uplimit[i_Reaction_serials]
        print('cluster_name_in_excel:', cluster_name)
        row = []
        if cluster_name not in All_cluster_name_in_files:
            no_log_and_out_files_of_cluster.append(cluster_name)
            continue

        row.append(i_Reaction_serials + 1)
        row.append(str(dic_cluster_serials[cluster_name][0]))
        row.append(dic_cluster_name[cluster_name][0])
        row.append(Molecule)

        # Molecule Cn, polar, IP,
        # row.append(Cn[Molecule])
        row.append(Molecule_polar[Molecule])
        row.append(Molecule_IP[Molecule])

        # Cluster charge
        row.append(dic_cluster_charge[cluster_name])

        # HOMO, LUMO, HOMO-LUMO gap Cluster
        row.append(dic_cluster_lumo[cluster_name][0])
        row.append(dic_cluster_homo[cluster_name][0])
        row.append(dic_cluster_homo_lumo_gap[cluster_name][0])

        # HOMO-LUMO gap between Cluster and Molecule
        E_LC_HM = dic_molecule_lumo[Molecule][0] - dic_cluster_homo[cluster_name][0]
        row.append(E_LC_HM)
        E_LM_HC = dic_cluster_lumo[cluster_name][0] - dic_molecule_homo[Molecule][0]
        row.append(E_LM_HC)

        # Max_Min_charge  of metal atoms
        for i in range(0, len(dic_max_min_charge_metal_atoms[cluster_name])):
            row.append(dic_max_min_charge_metal_atoms[cluster_name][i])

        # Dipole, Polar, Rotational_factor
        for i in range(0, len(dic_polar_dipole_rota_factor[cluster_name])):
            row.append(dic_polar_dipole_rota_factor[cluster_name][i])

        # Avg_NEC_s_d of metal atoms
        for i in range(0, len(dic_avg_s_d[cluster_name])):
            row.append(dic_avg_s_d[cluster_name][i])

        # VDE of Clusters
        for i in range(0, len(dic_VDE[cluster_name])):
            row.append(dic_VDE[cluster_name][i])

        # VEA of Clusters
        for i in range(0, len(dic_VEA[cluster_name])):
            row.append(dic_VEA[cluster_name][i][0])

        row.append(lgk1)
        row.append(Uplimit)

        csv_writer.writerow(row)
    csv_output.close()
    print('!!! These clusters have no log and out files:{}'.format(no_log_and_out_files_of_cluster))

if __name__ == "__main__":
    generate_result(df_cluster_in_excel, df_Molecule_in_excel, df_lgk1_in_excel, df_Uplimit)

