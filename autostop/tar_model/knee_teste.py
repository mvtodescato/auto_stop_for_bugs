# coding=utf-8

"""
The implementation is based on
[1] Satopaa, Ville, et al. "Finding a" kneedle" in a haystack: Detecting knee points in system behavior." 2011 31st international conference on distributed computing systems workshops. IEEE, 2011.
[2] Gordon V. Cormack and Maura R. Grossman. 2016. Engineering Quality and Reliability in Technology-Assisted Review. In Proceedings of the 39th International ACM SIGIR Conference on Research and Development in Information Retrieval (Pisa, Italy) (SIGIR ’16). ACM, New York, NY, USA, 75–84.
"""

import csv
import math
import numpy as np
from operator import itemgetter
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from autostop.tar_framework.assessing import Assessor
from autostop.tar_framework.ranking import Ranker
from autostop.tar_model.utils import *
from autostop.tar_framework.utils import *
from scipy.sparse import vstack
import random
import matplotlib.pyplot as plt
from scipy.integrate import simps

def detect_knee(data, window_size=1, s=10):
    """
    Detect the so-called knee in the data.

    The implementation is based on paper [1] and code here (https://github.com/jagandecapri/kneedle).

    @param data: The 2d data to find an knee in.
    @param window_size: The data is smoothed using Gaussian kernel average smoother, this parameter is the window used for averaging (higher values mean more smoothing, try 3 to begin with).
    @param s: How many "flat" points to require before we consider it a knee.
    @return: The knee values.
    """

    data_size = len(data)
    data = np.array(data)

    if data_size == 1:
        return None

    # smooth
    smoothed_data = []
    for i in range(data_size):

        if 0 < i - window_size:
            start_index = i - window_size
        else:
            start_index = 0
        if i + window_size > data_size - 1:
            end_index = data_size - 1
        else:
            end_index = i + window_size

        sum_x_weight = 0
        sum_y_weight = 0
        sum_index_weight = 0
        for j in range(start_index, end_index):
            index_weight = norm.pdf(abs(j-i)/window_size, 0, 1)
            sum_index_weight += index_weight
            sum_x_weight += index_weight * data[j][0]
            sum_y_weight += index_weight * data[j][1]

        smoothed_x = sum_x_weight / sum_index_weight
        smoothed_y = sum_y_weight / sum_index_weight

        smoothed_data.append((smoothed_x, smoothed_y))

    smoothed_data = np.array(smoothed_data)

    # normalize
    normalized_data = MinMaxScaler().fit_transform(smoothed_data)

    # difference
    differed_data = [(x, y-x) for x, y in normalized_data]

    # find indices for local maximums
    candidate_indices = []
    for i in range(1, data_size-1):
        if (differed_data[i-1][1] < differed_data[i][1]) and (differed_data[i][1] > differed_data[i+1][1]):
            candidate_indices.append(i)

    # threshold
    step = s * (normalized_data[-1][0] - data[0][0]) / (data_size - 1)

    # knees
    knee_indices = []
    for i in range(len(candidate_indices)):
        candidate_index = candidate_indices[i]

        if i+1 < len(candidate_indices):  # not last second
            end_index = candidate_indices[i+1]
        else:
            end_index = data_size

        threshold = differed_data[candidate_index][1] - step

        for j in range(candidate_index, end_index):
            if differed_data[j][1] < threshold:
                knee_indices.append(candidate_index)
                break

    if knee_indices != []:
        return knee_indices #data[knee_indices]
    else:
        return None


def test_detect_knee():
    # data with knee at [0.2, 0.75]
    print('First example.')
    data = [[0, 0],
            [0.1, 0.55],
            [0.2, 0.75],
            [0.35, 0.825],
            [0.45, 0.875],
            [0.55, 0.9],
            [0.675, 0.925],
            [0.775, 0.95],
            [0.875, 0.975],
            [1, 1]]

    knees = detect_knee(data, window_size=1, s=1)
    for knee in knees:
        print(data[knee])

    # data with knee at [0.45  0.1  ], [0.775 0.2  ]
    print('Second example.')
    data = [[0, 0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.35, 0.1],
            [0.45, 0.1],
            [0.55, 0.1],
            [0.675, 0.2],
            [0.775, 0.2],
            [0.875, 0.2],
            [1, 1]]
    knees = detect_knee(data, window_size=1, s=1)
    for knee in knees:
        print(data[knee])


def knee_method(data_name, topic_set, topic_id,
                query_file, qrel_file, doc_id_file, doc_text_file, # data parameters
                stopping_beta, rho, data_test, train_percentage, stopping_percentage=None, stopping_recall=1.0,  # autostop parameters
                random_state=0):
    #rho,stopping_beta
    """
    Implementation of the Knee method.
    See
    @param data_name:
    @param topic_set:
    @param topic_id:
    @param stopping_beta: stopping_beta: only stop TAR process until at least beta documents had been screen
    @param stopping_percentage:
    @param stopping_recall:
    @param rho:
    @param random_state:
    @return:
    """
    np.random.seed(random_state)
    list_recall = [0.0, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886, 0.03773584905660377, 0.03773584905660377, 0.05660377358490566, 0.05660377358490566, 0.07547169811320754, 0.07547169811320754, 0.07547169811320754, 0.09433962264150944, 0.16981132075471697, 0.20754716981132076, 0.2830188679245283, 0.33962264150943394, 0.41509433962264153, 0.5094339622641509, 0.6037735849056604, 0.6981132075471698, 0.8113207547169812, 0.8679245283018868, 0.8867924528301887, 0.8867924528301887, 0.8867924528301887, 0.9056603773584906, 0.9245283018867925, 0.9433962264150944, 0.9433962264150944, 0.9811320754716981, 0.9811320754716981, 1.0]
    list_sample = [0.00042498937526561835, 0.0012749681257968552, 0.0025499362515937103, 0.004249893752656183, 0.006374840628984276, 0.008924776880577986, 0.011899702507437314, 0.01529961750956226, 0.019124521886952826, 0.02337441563960901, 0.02804929876753081, 0.03357416064598385, 0.03994900127496812, 0.04717382065448364, 0.055248618784530384, 0.06417339566510838, 0.0743731406714832, 0.08584785380365491, 0.09859753506162346, 0.11262218444538886, 0.12834679133021676, 0.1457713557161071, 0.16532086697832554, 0.18699532511687209, 0.21121971950701232, 0.23799405014874628, 0.26774330641733957, 0.3004674883127922, 0.33659158521036975, 0.3765405864853379, 0.4207394815129622, 0.4696132596685083, 0.5235869103272418, 0.5830854228644284, 0.6485337866553336, 0.7207819804504887, 0.8002549936251594, 0.8878028049298767, 0.9842753931151721, 1.0]


    # model named with its configuration
    model_name = 'knee' + '-'
    model_name += 'sb' + str(stopping_beta) + '-'
    model_name += 'sp' + str(stopping_percentage) + '-'
    model_name += 'sr' + str(stopping_recall) + '-'

    model_name += 'rho' + str(rho) + '-'
    LOGGER.info('Model configuration: {}.'.format(model_name))

    # loading data
    datamanager = Assessor(query_file, qrel_file, doc_id_file, doc_text_file)
    query_file2 = os.path.join(PARENT_DIR, 'data', data_test, 'topics', '1')
    qrel_file2 = os.path.join(PARENT_DIR, 'data', data_test, 'qrels', '1')
    doc_id_file2 = os.path.join(PARENT_DIR, 'data', data_test, 'docids', '1')
    doc_text_file2 = os.path.join(PARENT_DIR, 'data', data_test, 'doctexts', '1')
    datamanager2 = Assessor(query_file2, qrel_file2, doc_id_file2, doc_text_file2)
    complete_dids = datamanager.get_complete_dids()
    total = datamanager.get_total_rel_num()
    test_dids = datamanager2.get_complete_dids()
    complete_pseudo_dids = datamanager.get_complete_pseudo_dids()
    did2label = datamanager2.get_did2label()
    total_true_r = datamanager2.get_total_rel_num()
    total_num = datamanager2.get_total_doc_num()
    # preparing document features
    ranker = Ranker()
    ranker.set_did_2_feature(dids=complete_pseudo_dids, data_name = data_name)
    ranker.set_features_by_name('complete_dids', complete_dids)
    ranker2 = Ranker()
    ranker2.set_did_2_feature(dids=test_dids, data_name = data_test)
    ranker2.set_features_by_name('test_dids', test_dids)
    # local parameters
    stopping = False
    t = 0
    batch_size = 1
    temp_doc_num = 100
    knee_data = []
    #tam=int(total * train_percentage)
    #x = random.sample(range(total), tam)
    #y = np.sort(x)
    #z = y.tolist()
    name_train = ''
    if train_percentage > 1:
        num_treino = train_percentage
        name_train = str(train_percentage) + ' arq. treino'
    else:
        if type(train_percentage) == int:
            num_treino = train_percentage
            name_train = str(train_percentage) + ' arq. treino'
        else:
            num_treino = int(total * train_percentage)
            name_train = str(100/train_percentage) + '% ' +'arq. treino'
        print(num_treino)
        #levar em consideração a quantidade de relevantes total.
    


    recall = []
    sampled = []

    # starting the TAR process
    interaction_file = name_interaction_file(data_name=data_test, model_name=model_name, topic_set=topic_set,
                                             exp_id=random_state, topic_id=topic_id)
    with open(interaction_file, 'w', encoding='utf8') as f:
        csvwriter = csv.writer(f)
        while not stopping:
            t += 1
            LOGGER.info('TAR: iteration={}'.format(t))
            LOGGER.info('Train={}, Teste:{}, % = {}'.format(data_name,data_test,train_percentage))
            #ponto importante
            train_dids1, train_labels1 = datamanager.get_training_data4(num_treino)
            print(len(train_dids1))
            train_dids2, train_labels2 = datamanager2.get_training_data(temp_doc_num)
            train_labels = train_labels1 + train_labels2
            train_features1 = ranker.get_feature_by_did(train_dids1)
            #print(train_features1)
            #print(len(train_labels))
            train_features2 = ranker2.get_feature_by_did(train_dids2)
            train_features = vstack([train_features1,train_features2])
            #print(train_features)
            ranker.train(train_features, train_labels)
            test_features = ranker2.get_features_by_name('test_dids')
            scores = ranker.predict(test_features)
            #print(scores)
            zipped = sorted(zip(test_dids, scores), key=itemgetter(1), reverse=True)
            ranked_dids, score = zip(*zipped)
            #print(ranked_dids)
            #print(score)

            # cutting off instead of sampling
            selected_dids = datamanager2.get_top_assessed_dids(ranked_dids, batch_size)
            #print(selected_dids)
            datamanager2.update_assess(selected_dids)

            # statistics
            sampled_num = datamanager2.get_assessed_num()
            #print(sampled_num)
            sampled_percentage = sampled_num/total_num
            running_true_r = datamanager2.get_assessed_rel_num()
            #print(running_true_r)
            #break
            running_true_recall = running_true_r / float(total_true_r)
            ap = calculate_ap(did2label, ranked_dids)

            # update parameters
            batch_size += math.ceil(batch_size / 10)

            # debug: writing values
            csvwriter.writerow(
                (t, batch_size, total_num, sampled_num, total_true_r, running_true_r, ap, running_true_recall))
            recall.append(running_true_recall)
            sampled.append(sampled_percentage)
            # detect knee
            knee_data.append((sampled_num, running_true_r))
            knee_indice = detect_knee(knee_data)  # x: sampled_percentage, y: running_true_r
            if knee_indice is not None:

                knee_index = knee_indice[-1]
                rank1, r1 = knee_data[knee_index]
                rank2, r2 = knee_data[-1]

                try:
                    current_rho = float(r1 / rank1) / float((r2 - r1 + 1) / (rank2 - rank1))
                except:
                    print('(rank1, r1) = ({} {}), (rank2, r2) = ({} {})'.format(rank1, r1, rank2, r2))
                    current_rho = 0  # do not stop

                if rho == 'dynamic':
                    rho = 156 - min(running_true_r, 150)   # rho is in [6, 156], see [1]
                else:
                    rho = float(rho)

                #if current_rho > rho:
                #    if sampled_num > stopping_beta:
                #        stopping = True

            # debug: stop early
            if stopping_recall:
                if running_true_recall >= stopping_recall:
                    stopping = True
            if stopping_percentage:
                if sampled_num >= int(total_num * stopping_percentage):
                    stopping = True
            if sampled_num == total_num:
                stopping = True
    
    y_calc = np.array(recall)
    area = simps(y_calc, dx=5)
    print("area =", area)
    plt.plot(sampled, recall, label=data_name + "(treino: " + str(name_train) + ")")
    plt.suptitle(data_name + "(treino) | " + data_test + "(teste)")
    plt.title("Recall x Custo")
    
    plt.grid(True)
    plt.xlabel("Custo de rotulação")
    plt.ylabel("Recall")
    plt.legend()
    if train_percentage == 1.0 and type(train_percentage) == float:
        plt.plot(list_sample, list_recall, label="Original",color='black', linewidth=1.3)
        plt.suptitle(data_name + "(treino) | " + data_test + "(teste)")
        plt.title("Recall x Custo")
        plt.grid(True)
        plt.xlabel("Custo de rotulação")
        plt.ylabel("Recall")
        plt.legend()
        plt.savefig(interaction_file[0:-4] + '_graph.pdf')
        plt.clf()
    
    
    
    shown_dids = datamanager2.get_assessed_dids()
    check_func = datamanager2.assess_state_check_func()
    tar_run_file = name_tar_run_file(data_name=data_test, model_name=model_name, topic_set=topic_set,
                                     exp_id=random_state, topic_id=topic_id)
    with open (tar_run_file[0:-4] + "_area","w") as area_arq:
        area_arq.write(str(area))
    
    with open(tar_run_file, 'w', encoding='utf8') as f:
        write_tar_run_file(f=f, topic_id=topic_id, check_func=check_func, shown_dids=shown_dids)

    LOGGER.info('TAR is finished.')

    return


def main(rho,stopping_beta,topic,data_train,data_test,train_percentage):
    data_name = data_train
    topic_id = topic
    topic_set = data_train + '(train' + str(train_percentage) + ')' + data_test + '(test)'
    query_file = os.path.join(PARENT_DIR, 'data', data_name, 'topics', topic_id)
    qrel_file = os.path.join(PARENT_DIR, 'data', data_name, 'qrels', topic_id)
    doc_id_file = os.path.join(PARENT_DIR, 'data', data_name, 'docids', topic_id)
    doc_text_file = os.path.join(PARENT_DIR, 'data', data_name, 'doctexts', topic_id)

    knee_method(data_name, topic_id, topic_set,query_file, qrel_file, doc_id_file, doc_text_file,stopping_beta,rho,data_test,train_percentage)


#main(rho=6,stopping_beta=1000,topic='1',data_train='android',data_test='mct',train_percentage=1)