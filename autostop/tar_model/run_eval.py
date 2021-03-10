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
            if 'rho' + str(rho) in f:
                if 'sb' + str(beta) in f:
                    md_name = f
                    print("sim")
        
        path = retdir + '/' + data_name + '/' + 'tar_run' + '/' + md_name + '/' + exp_id + '/' + zero + '/' + data_name + '.run'
        print(path)
    else:
        path = retdir + '/' + data_name + '/' + 'tar_run' + '/' + model_name + '/' + exp_id + '/' + zero + '/' + data_name + '.run'
    print(path)
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
    
    #print(ret)
    ret = subprocess.check_output([' tail -27 '], shell=True, input=ret)
    ret = ret.decode(encoding='utf-8')
    #print(ret)
   
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
    #do 1 ao 10 com 6
    #testar o rho dinamico e ver o valor que ele gera
    #deixar o beta apenas em 100
    #plotar grafico de linha, porcentagem de custo e recall
    for data in datas:
        for rho in['dynamic', 10/6, 12/8, 0.01, 5, 8, 10 , 15]:
            for beta in[100.0, 1000.0]:
                knee.main(rho=rho, stopping_beta=beta,topic=topic,data=data)
    dfs = []
    for data in datas:
        for rho in['dynamic', 10/6, 12/8, 0.01, 5, 8, 10 , 15]:
            for beta in[100.0, 1000.0]:
                _df = TarEvalResultReader(data_name=data, model_name= 'knee_sb' + str(beta) + '-sp1.0-srNone-rho' + str(rho)  , exp_id='1', train_test=data, topic_id=topic,zero="0",rho=rho,beta=beta,method_name='knee') #o zero na vdd tem relação com o random state, se ele for 1 o zero tem q ser 1 tbm
                _df['dta'] = data
                _df['rho'] = rho
                _df['beta'] = beta
                dfs.append(_df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby(['dta','bound', 'beta']).mean()

    df[['cost', 'recall','loss_er']]
    df.to_csv(r'/home/mvtodescato/auto-stop-tar/autostop/tar_model/results_csv/knee_exec.csv')
    print(df)

def scal_exec(topic,datas):
    dfs = []
    for data in datas:
        for sub_percentage in [0.8, 1.0]:
            for bound_bt in [30, 50, 70, 90, 110]:
                for ita in [1.0, 1.05]:
                    scal.main(sub_percentage, bound_bt, ita, topic,data)
    for data in datas:
        for sub_percentage in [0.8, 1.0]:
            for bound_bt in [30, 50, 70, 90, 110]:
                for ita in [1.0, 1.05]:
                    _df = TarEvalResultReader(data_name=data, model_name='scal-sp1.0-sr1.0-tr1.0-spt{}-bnd{}-mxnmin-bktsamplerel-ita{}'.format(sub_percentage, bound_bt, ita), exp_id='1', train_test=data, topic_id=topic, zero="0",rho=0,beta=0, method_name='scal')
                    _df['dta'] = data
                    _df['spt'] = sub_percentage
                    _df['bnd'] = bound_bt
                    _df['ita'] = ita
                    dfs.append(_df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby(['dta','spt', 'bnd', 'ita']).mean()
    df[['wss_100', 'loss_er', 'norm_area', 'recall', 'cost', 'num_shown', 'num_docs', 'rels_found', 'num_rels', 'ap', 'NCG@10', 'NCG@100']]
    df.to_csv(r'/home/mvtodescato/auto-stop-tar/autostop/tar_model/results_csv/scal_exec.csv')
    print(df)

def autostop_exec(topic,datas):
    dfs = []
    for data in datas:
        for target_recall in [1.0, 0.9, 0.8]:
            for sampler_type in ['HTAPPriorSampler','HTUniformSampler','HTPowerLawSampler','HHPowerLawSampler','HHAPPriorSampler']:
                for stop_condition in ['strict1','loose','strict2']:
                    auto_stop.main(target_recall, sampler_type, stop_condition, topic, data)
    for data in datas:
        for target_recall in [1.0, 0.9, 0.8]:
            for sampler_type in ['HTAPPriorSampler','HTUniformSampler','HTPowerLawSampler','HHPowerLawSampler','HHAPPriorSampler']:
                for stop_condition in ['strict1','loose','strict2']:
                    _df = TarEvalResultReader(data_name=data, model_name='autostop-spNone-sr1.0-smp{}-tr{}-sc{}'.format(sampler_type, target_recall, stop_condition), exp_id='1', train_test=data, topic_id=topic, zero="0",rho=0,beta=0, method_name='autostop')
                    _df['dta'] = data
                    _df['smp'] = sampler_type
                    _df['sc'] = stop_condition
                    _df['tr'] = target_recall
                    dfs.append(_df)
                
    df = pd.concat(dfs, ignore_index=True)
    df['reliability'] = df.apply(lambda row:1 if row['recall'] >= row['tr'] else 0, axis=1)
    df['re'] = np.abs(df['recall'] - df['tr']) / df['tr']
    df = df.groupby(['dta', 'smp', 'sc', 'tr']).mean()
    df.to_csv(r'/home/mvtodescato/auto-stop-tar/autostop/tar_model/results_csv/autostop_exec.csv')
    print(df)

import knee
import tar_eval
import scal
import auto_stop
topic = "1"
datas = ['elasticsearch'] #,'hazelcast','neo4j']

knee_exec(topic, datas)

scal_exec(topic, datas)

autostop_exec(topic, datas)