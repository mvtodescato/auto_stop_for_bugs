#knee
import re
import os
import math
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
pd.set_option('precision', 3)
import subprocess
import csv
from collections import  defaultdict
from operator import itemgetter
from scipy import interpolate

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")


show_columns = ['recall', 'cost', 'reliability', 'loss_er', 're']

tar_master_dir = '/home/mvtodescato/auto-stop-tar/autostop/tar_model'
datadir = '/home/mvtodescato/auto-stop-tar/data'
retdir = '/home/mvtodescato/auto-stop-tar/ret'

def TarEvalResultReader(data_name, model_name, exp_id, train_test, topic_id,zero,rho,beta,method_name):
    
    # run file
    #os.path.join(retdir, data_name, 'tar_run', model_name, exp_id, train_test, topic_id + '.run')
    filedirlist = os.listdir('/home/mvtodescato/auto-stop-tar/ret' +'/'+ data_name + '/' + 'tar_run')
    if method_name == 'knee':
        for f in filedirlist:
            print(f)
            print(model_name)
            if 'rho' + str(beta) in f:
                if 'sb' + str(rho) in f:
                    md_name = f
                    print("sim")
        
        path = retdir + '/' + data_name + '/' + 'tar_run' + '/' + md_name + '/' + exp_id + '/' + zero + '/' + data_name + '.run'
        print(path)
    else:
        path = retdir + '/' + data_name + '/' + 'tar_run' + '/' + model_name + '/' + exp_id + '/' + zero + '/' + data_name + '.run'
    #path2 = '/home/mvtodescato/auto-stop-tar/ret/{}/tar_run/{}/1/0/{}.run'.format(data_name,model_name,data_name)
    #path2 = path2.lstrip("/")
    runfile = os.path.join(path)
                            
    # qrel file
    #print(runfile)
    qrelfile = os.path.join(datadir, data_name, 'qrels', topic_id)
    print(qrelfile)
    # tar eval script
    script = os.path.join(tar_master_dir, 'tar_eval.py')
    
    # result
    #ret = tar_eval.main(qrel=qrelfile, rel=runfile)
    ret = subprocess.check_output(['python3', script, qrelfile, runfile])
    
    print(ret)
    ret = subprocess.check_output([' tail -27 '], shell=True, input=ret)
    ret = ret.decode(encoding='utf-8')
    print(ret)
   
    # dataframe
    dct = {}
    for line in ret.split('\n'):
      if line != '':
            tid, key, val = line.split()
            if tid == 'ALL':
                dct[key] = [float(val)]
    
    df = pd.DataFrame(dct)
    model = model_name
    df['model_name'] = [model]
    df['exp_id'] = [exp_id]
    df['topic_id'] = [topic_id]
    df['recall'] = float(df['rels_found']) / float(df['num_rels'])
    df['cost'] = float(df['num_shown']) / float(df['num_docs'])
    return df


def knee_exec(topic,datas):
#    for data in datas:
#        for rho in['dynamic', 10/6, 12/8, 0.01, 5, 8, 10 , 15]:
#            for beta in[100.0, 1000.0]:
#                knee.main(rho=rho, stopping_beta=beta,topic=topic,data=data)
    dfs = []
    for data in datas:
        for rho in['dynamic', 10/6, 12/8, 0.01, 5, 8, 10 , 15]:
            for beta in[100.0, 1000.0]:
                _df = TarEvalResultReader(data_name=data, model_name= 'knee_sb' + str(rho) + '-sp1.0-srNone-rho' + str(beta), exp_id='1', train_test=data, topic_id=topic,zero="0",rho=rho,beta=beta,method_name='knee') #o zero na vdd tem relação com o random state, se ele for 1 o zero tem q ser 1 tbm
                _df['bound'] = rho
                _df['beta'] = beta
                dfs.append(_df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby(['bound', 'beta']).mean()

    df[['cost', 'recall','loss_er']]

    print(df)

def scal_exec(topic,datas):
    dfs = []
    #for data in datas:
    #    for sub_percentage in [0.8, 1.0]:
    #        for bound_bt in [30, 50, 70, 90, 110]:
    #            for ita in [1.0, 1.05]:
    #                scal.main(sub_percentage, bound_bt, ita, topic,data)
    for data in datas:
        for sub_percentage in [0.8, 1.0]:
            for bound_bt in [30, 50, 70, 90, 110]:
                for ita in [1.0, 1.05]:
                    _df = TarEvalResultReader(data_name=data, model_name='scal-sp1.0-sr1.0-tr1.0-spt{}-bnd{}-mxnmin-bktsamplerel-ita{}'.format(sub_percentage, bound_bt, ita), exp_id='1', train_test=data, topic_id=topic, zero="0",rho=0,beta=0, method_name='scal')
                    _df['spt'] = sub_percentage
                    _df['bnd'] = bound_bt
                    _df['ita'] = ita
                    dfs.append(_df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby(['spt', 'bnd', 'ita']).mean()
    df[['wss_100', 'loss_er', 'norm_area', 'recall', 'cost', 'num_shown', 'num_docs', 'rels_found', 'num_rels', 'ap', 'NCG@10', 'NCG@100']]
    print(df)

import knee
import tar_eval
import scal
topic = "1"
datas = ['anttlr4']

#knee_exec(topic=topic, datas=datas)

scal_exec(topic=topic, datas=datas)

#with open('employee_file.csv', mode='w') as employee_file:
#    employee_file = csv.writer(df)