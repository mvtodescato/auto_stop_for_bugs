import os
import pandas as pd
pasta = "./GitHubBugDataSet-1.1/database"
caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
#rquivos = [arq for arq in caminhos if os.path.isfile(arq)]
#print(arquivos)
for past in caminhos:
    caminho = [os.path.join(past, nome) for nome in os.listdir(past)]
    for cam in caminho:
        c = [os.path.join(cam, nome) for nome in os.listdir(cam)]
        data = pd.read_csv(c[0])
        data_conc 