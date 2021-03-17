import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
    


def make_matrix(data_name, len_ids):    
    matriz = csr_matrix((len_ids,107))
    with open("../../data/" + data_name + "/" + data_name + ".svm.fil", "r") as svm:
                for line in svm:
                    new_line = line.split()
                    for var in new_line:
                        if var == new_line[0]:
                            x = var
                            continue
                        n_line = var.split(":")
                        #print(x)
                        #print(n_line[0])
                        matriz[int(x),int(n_line[0])-1] = n_line[1]

    print(matriz)
    scipy.sparse.save_npz("../../data/" + data_name + "/" + data_name + '.npz', matriz)

datas = ['oryx','titan','ceylon','hazelcast','broadleaf']
len_ids = [2157,5312,4512,25130,17433]
i=0
for data in datas:
    make_matrix(data,len_ids[i])
    i = i + 1
    print("feito " + data)

