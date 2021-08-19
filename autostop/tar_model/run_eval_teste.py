# sknee
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
from collections import defaultdict
from operator import itemgetter
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

show_columns = ['recall', 'cost', 'reliability', 'loss_er', 're']

tar_master_dir = '/home/mvtodescato/auto_stop_for_bugs/autostop/tar_model'
datadir = '/home/mvtodescato/auto_stop_for_bugs/data'
retdir = '/home/mvtodescato/auto_stop_for_bugs/ret'
global area

def TarEvalResultReader(
        data_name, model_name, exp_id,
        topic_id, zero, rho, beta, method_name,all_name):

    # run file
    # os.path.join(
    #        retdir, data_name, 'tar_run', model_name,
    #        exp_id, train_test, topic_id + '.run')

    filedirlist = os.listdir(retdir + '/' + data_name + '/' + 'tar_run')
    if method_name == 'knee':
        for f in filedirlist:
            print(f)
            print(model_name)
            if 'rho' + str(rho) + '-' in f:
                if 'sb' + str(beta) + '-' in f:
                    md_name = f

        path = retdir + '/' + data_name + '/' + 'tar_run' + '/' + md_name + '/' + exp_id + '/' + zero + '/' + all_name + '.run'
        print(path)
    else:
        path = retdir + '/' + data_name + '/' + 'tar_run' + '/' + model_name + '/' + exp_id + '/' + zero + '/' + all_name + '.run'
    print(path)
    # path2 = '/home/mvtodescato/auto-stop-tar/ret/{}/tar_run/{}/1/0/{}.run'.format(data_name,model_name,data_name)
    # path2 = path2.lstrip("/")
    runfile = os.path.join(path)

    # qrel file
    # print(runfile)
    qrelfile = os.path.join(datadir, data_name, 'qrels', topic_id)
    print(qrelfile)
    # tar eval script
    script = os.path.join(tar_master_dir, 'tar_eval.py')

    # result
    # ret = tar_eval.main(qrel=qrelfile, rel=runfile)
    ret = subprocess.check_output(['python3', script, qrelfile, runfile])

    # print(ret)
    ret = subprocess.check_output([' tail -27 '], shell=True, input=ret)
    ret = ret.decode(encoding='utf-8')
    # print(ret)

    # dataframe
    dct = {}
    for line in ret.split('\n'):
        if line != '':
            tid, key, val = line.split()
            if tid == 'ALL':
                dct[key] = [float(val)]
    
    with open (path[0:-4] + "_area","r") as area_arq:
        area = area_arq.readline()
    

    df = pd.DataFrame(dct)
    model = model_name
    df['model_name'] = [model]
    df['exp_id'] = [exp_id]
    df['topic_id'] = [topic_id]
    df['recall'] = float(df['rels_found']) / float(df['num_rels'])
    df['cost'] = float(df['num_shown']) / float(df['num_docs'])
    return df , area


def knee_exec(topic, datas, datas_train):
    for data_train in datas_train:
        for data_test in datas:
            if data_train == data_test:
                continue
            for rho in [6]:
                for beta in [1000.0]:
                    if data_test == 'android':
                        beta = 100
                    
                    for train_percentage in [1,5,10,1.0]:
                        knee_teste.main(rho, beta, topic, data_train,data_test,train_percentage)
    dfs = []
    for data_train in datas_train:
        for data_test in datas:
            if data_train == data_test:
                continue
            for rho in [6]:
                for beta in [1000.0]:
                    if data_test == 'android':
                        beta = 100
                    for train_percentage in [1,5,10,1.0]:
                        m_name = 'knee_sb' + str(beta)+ '-spNone-sr1.0-rho' + str(rho)
                        t_train_percentage = str(train_percentage)
                        # o zero na vdd tem relação com o random state,
                        # se ele for 1 o zero tem q ser 1 tbm
                        _df , area = TarEvalResultReader(
                                data_name=data_test, model_name=m_name, exp_id='1',
                                topic_id=topic, zero='0',
                                rho=rho, beta=beta, method_name='knee',all_name=data_train + '(train' + t_train_percentage + ')' + data_test + '(test)')
                        _df['dta_train'] = data_train
                        _df['train_percentage'] = train_percentage
                        _df['dta_test'] = data_test
                        _df['rho'] = rho
                        _df['beta'] = beta
                        area = float(area)
                        _df['area'] = "{:.2f}".format(area)
                        dfs.append(_df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby(['dta_train', 'dta_test','train_percentage','area']).mean()

    df[['cost', 'recall', 'loss_er']]
    df.to_csv(tar_master_dir + '/results_csv/knee_exec_all.csv')
    print(df)
    

def scal_exec(topic, datas):
    dfs = []
    for data_train in datas:
        for data_test in datas:
            if data_train == data_test:
                continue
            if data_test != 'junit':
                if data_test != 'mct':
                    continue
            for sub_percentage in [1.0]:
                for bound_bt in [110]:
                    for ita in [1.05]:
                        for train_percentage in [0.1]:
                            scal_teste.main(sub_percentage, bound_bt, ita, topic, data_train,data_test,train_percentage)
    for data_train in datas:
        for data_test in datas:
            if data_train == data_test:
                continue
            if data_test != 'junit':
                if data_test != 'mct':
                    continue
            for sub_percentage in [1.0]:
                for bound_bt in [110]:
                    for ita in [1.05]:
                        for train_percentage in [0.1]:
                            m_name = 'scal-sp1.0-sr1.0-tr1.0-spt{}-bnd{}-mxnmin-bktsamplerel-ita{}'.format(
                                    sub_percentage, bound_bt, ita)
                            t_train_percentage = str(train_percentage)
                            print(t_train_percentage)
                            _df = TarEvalResultReader(
                                    data_name=data_test, model_name=m_name, exp_id='1',
                                    topic_id=topic, zero='0',
                                    rho=0, beta=0, method_name = 'scal',all_name=data_train + '(train' + t_train_percentage + ')' + data_test + '(test)')
                            _df['dta_train'] = data_train
                            _df['train_percentage'] = train_percentage
                            _df['dta_test'] = data_test
                            _df['spt'] = sub_percentage
                            _df['bnd'] = bound_bt
                            _df['ita'] = ita
                            dfs.append(_df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby(['dta_train', 'dta_test','train_percentage']).mean()
    df[
        [
            'wss_100',
            'loss_er',
            'norm_area',
            'recall',
            'cost',
            'num_shown',
            'num_docs',
            'rels_found',
            'num_rels',
            'ap',
            'NCG@10',
            'NCG@100',
        ]
    ]
    df.to_csv(tar_master_dir + '/results_csv/scal_exec_10_100random.csv')
    print(df)

if __name__ == '__main__':
    import knee_teste
    import tar_eval
    import scal_teste
    topic = '1'
    datas_train = ['android','elasticsearch'] #'elasticsearch']     ##,
    datas = ['anttlr4']#,'junit','mct']

    knee_exec(topic, datas,datas_train)

    #scal_exec(topic, datas)
