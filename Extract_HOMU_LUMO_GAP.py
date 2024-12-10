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


def check_error(path):
    file = open(path, "r").readlines()
    if "Normal termination" in file[-20:]:
        return "Normal"
    else:
        return "Termination Error"

    pattern = r"(Frequencies --)( +)(-?(\d+(\.\d+)?))"
    result = re.search(pattern, file)
    if result is not None:
        result = result.group(3)
        frequency = float(result)
        if frequency < 0:
            return "Frequency Error: " + result
        else:
            return "No imaginary frequency"
    else:
        return "No frequency"


def file_processor(path):
    print(path)
    block='Population analysis'
    eigen='eigenvalues'
    Alpha_occst='Alpha  occ. eigenvalues'
    Alpha_virst='Alpha virt. eigenvalues'
    Beta_occst ='Beta  occ. eigenvalues'
    Beta_virst ='Beta virt. eigenvalues'
    sepst='--'
    file = open(path, "r")
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
    
    return 	Alpha_HOMO,Alpha_LUMO,Beta_HOMO,Beta_LUMO


def generate_result():
    result_path = get_all_file()
    # result_output = open("result_output.txt", "w")
    csv_output = open("MO.csv", "w")
    # result_output.close()
    csv_output.close()
    # result_output = open("result_output.txt", "a+")
    csv_output = open("MO.csv", "a+", newline="", encoding="utf-8")
    header = ["Cluster-name",  "Alpha_HOMO (eV)", "Alpha_LUMO (eV)", "Alpha_HOMO-LUMO gap (eV)", "Beta_HOMO (eV)", "Beta_LUMO (eV)", "Beta_HOMO-LUMO gap (eV)","which LUMO in H-L gap","which HOMO in H-L gap", "H-L gap"]
    csv_writer = csv.writer(csv_output, dialect='excel')
    csv_writer.writerow(header)

    for path in result_path:
        row = []
        dir_name, file_name = os.path.split(path)
        file_name = file_name.split('.')[0]
        cluster = file_name.split('-')[0]
        multi = file_name.split('-')[2]
        num = file_name.split('-')[3]
        cluster_name = cluster + '-' + multi + '-' + num
        row.append(cluster_name)

        error = check_error(path)
        result = file_processor(path)
        
        Alpha_HOMO = float(result[0]) * 27.2113682
        Alpha_LUMO = float(result[1]) * 27.2113682
        row.append(Alpha_HOMO)
        row.append(Alpha_LUMO)
        Alpha_HOMO_LUMO_gap = (Alpha_LUMO-Alpha_HOMO)
        row.append(Alpha_HOMO_LUMO_gap)
        
        Beta_HOMO = float(result[2]) * 27.2113682
        Beta_LUMO = float(result[3]) * 27.2113682
        row.append(Beta_HOMO)
        row.append(Beta_LUMO)
        Beta_HOMO_LUMO_gap = (Beta_LUMO-Beta_HOMO)
        row.append(Beta_HOMO_LUMO_gap)

        if Alpha_LUMO>Beta_LUMO:
            Lumo = Beta_LUMO
            row.append('Beta_LUMO')
        else:
            Lumo = Alpha_LUMO
            row.append('Alpha_LUMO')

        if Alpha_HOMO>Beta_HOMO:
            Homo = Alpha_HOMO
            row.append('Alpha_HOMO')
        else:
            Homo = Beta_HOMO
            row.append('Beta_HOMO')

        H_L_gap = Lumo - Homo
        row.append(H_L_gap)
        
        generated_error = ""        
        if "Termination" in error or "Frequency" in error:
            # result_output.write(error + "\n")
            generated_error += error
        #row.append(generated_error)
        csv_writer.writerow(row)
    # result_output.close()
    csv_output.close()


if __name__ == "__main__":
    generate_result()
