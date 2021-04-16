#pyh# coding=utf-8

"""
The implementation is based on
[1] Gordon V. Cormack and Maura R. Grossman. 2016. Scalability of Continuous Active Learning for Reliable High- Recall Text Classification. In Proceedings of the 25th ACM International on Conference on Information and Knowledge Management (Indianapolis, Indiana, USA) (CIKM ’16). ACM, New York, NY, USA, 1039–1048.
"""
import csv
import math
import numpy as np
import random
from operator import itemgetter
from autostop.tar_framework.assessing import Assessor
from autostop.tar_framework.ranking import Ranker
from autostop.tar_framework.sampling_estimating import SCALSampler
from autostop.tar_model.utils import *
from autostop.tar_framework.utils import *
from scipy.sparse import vstack

def scal_method(data_name, topic_set, topic_id,
                query_file, qrel_file, doc_id_file, doc_text_file, sub_percentage, bound_bt, ita, data_test,train_percentage, # data parameters
                stopping_percentage=1.0, stopping_recall=1.0, target_recall=1.0,  # autostop parameters
                max_or_min='min', bucket_type='samplerel',  # scal parameters
                random_state=0):
    data_name, topic_id, topic_set, query_file, qrel_file, doc_id_file, doc_text_file,sub_percentage,bound_bt,ita
    """
    Implementation of the S-CAL method [1].

    @param data_name:
    @param topic_set:
    @param topic_id:
    @param stopping_percentage:
    @param stopping_recall:
    @param target_recall:
    @param sub_percentage:
    @param bound_bt:  sample size per batch
    @param max_or_min:
    @param bucket_type:
    @param ita:
    @param random_state:
    @return:
    """
    np.random.seed(random_state)

    # model named with its configuration
    model_name = 'scal' + '-'
    model_name += 'sp' + str(stopping_percentage) + '-'
    model_name += 'sr' + str(stopping_recall) + '-'
    model_name += 'tr' + str(target_recall) + '-'
    model_name += 'spt' + str(sub_percentage) + '-'
    model_name += 'bnd' + str(bound_bt) + '-'
    model_name += 'mxn' + max_or_min + '-'
    model_name += 'bkt' + bucket_type + '-'
    model_name += 'ita' + str(ita)
    LOGGER.info('Model configuration: {}.'.format(model_name))

    # loading data
    datamanager = Assessor(query_file, qrel_file, doc_id_file, doc_text_file)

    query_file2 = os.path.join(PARENT_DIR, 'data', data_test, 'topics', '1')
    qrel_file2 = os.path.join(PARENT_DIR, 'data', data_test, 'qrels', '1')
    doc_id_file2 = os.path.join(PARENT_DIR, 'data', data_test, 'docids', '1')
    doc_text_file2 = os.path.join(PARENT_DIR, 'data', data_test, 'doctexts', '1')
    datamanager2 = Assessor(query_file2, qrel_file2, doc_id_file2, doc_text_file2)

    complete_dids = datamanager.get_complete_dids()
    total = len(complete_dids)
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

    # SCAL sampler
    sampler = SCALSampler()

    # sampling a large sample set before the TAR process
    if sub_percentage < 1.0:
        u = int(sub_percentage * total_num)
    elif sub_percentage == 1.0:
        u = total_num
    else:
        raise NotImplementedError

    # local parameters
    stopping = False
    t = 0
    batch_size = 1
    temp_doc_num = 100
    n = bound_bt

    total_esti_r = 0
    temp_list = []
    tam=int(total * train_percentage)
    x = random.sample(range(total), tam)
    y = np.sort(x)
    z = y.tolist()
    # starting the TAR process
    interaction_file = name_interaction_file(data_name=data_test, model_name=model_name, topic_set=topic_set,
                                             exp_id=random_state, topic_id=topic_id)
    with open(interaction_file, 'w', encoding='utf8') as f:
        csvwriter = csv.writer(f)
        while not stopping:
            t += 1

            LOGGER.info('iteration {}, batch_size {},data_train {},data_test {},percentage {}'.format(t, batch_size,data_name,data_test,train_percentage))

            # train
            train_dids1, train_labels1 = datamanager.get_training_data2(z)
            train_dids2, train_labels2 = datamanager2.get_training_data(temp_doc_num)
            train_labels = train_labels1 + train_labels2
            train_features1 = ranker.get_feature_by_did(train_dids1)
            train_features2 = ranker2.get_feature_by_did(train_dids2)
            train_features = vstack([train_features1,train_features2])
            ranker.train(train_features, train_labels)

            # predict
            test_features = ranker2.get_features_by_name('test_dids')
            scores = ranker.predict(test_features)
            #print(scores)
            zipped = sorted(zip(test_dids, scores), key=itemgetter(1), reverse=True)
            ranked_dids, score = zip(*zipped)

            bucketed_dids,sampled_dids, batch_esti_r = sampler.sample(ranked_dids, n, batch_size, did2label)
            datamanager2.update_assess(sampled_dids)

            # estimating
            total_esti_r += batch_esti_r

            # statistics
            sampled_num = datamanager2.get_assessed_num()
            running_true_r = datamanager2.get_assessed_rel_num()
            if total_esti_r != 0:
                running_esti_recall = running_true_r / float(total_esti_r)
            else:
                running_esti_recall = 0
            if total_true_r != 0:
                running_true_recall = running_true_r / float(total_true_r)
            else:
                running_true_recall = 0
            ap = calculate_ap(did2label, ranked_dids)

            # update parameters
            if batch_size < total_num:  # avoid OverflowError
                batch_size += math.ceil(batch_size / 10)

            # debug: writing values
            csvwriter.writerow(
                (t, batch_size, total_num, sampled_num, total_true_r, total_esti_r, running_true_r, ap, running_esti_recall, running_true_recall))

            cum_bucketed_dids = sampler.get_bucketed_dids()
            cum_sampled_dids = sampler.get_sampled_dids()
            temp_list.append((total_esti_r, ranker, cum_bucketed_dids, cum_sampled_dids))

            # when sub sample is exhausted, stop
            len_bucketed_dids = len(cum_bucketed_dids)
            if len_bucketed_dids == u:
                stopping = True

            # debug: stop early
            if stopping_recall:
                if running_true_recall >= stopping_recall:
                    stopping = True
            if stopping_percentage:
                if sampled_num >= int(total_num * stopping_percentage):
                    stopping = True

    # estimating rho
    final_total_esti_r = ita * total_esti_r  # calibrating the estimation in [1]

    if max_or_min == 'max':
        max_or_min_func = max
    elif max_or_min == 'min':
        max_or_min_func = min
    else:
        raise NotImplementedError

    # finding the first ranker that satisfies the stopping strategy, otherwise using the last ranker
    for total_esti_r, ranker, bucketed_dids, cum_sampled_dids in temp_list:
        if target_recall * final_total_esti_r <= total_esti_r:
            break

    if bucket_type == 'bucket':
        filtered_dids = bucketed_dids
    elif bucket_type == 'sample':
        filtered_dids = cum_sampled_dids
    elif bucket_type == 'samplerel':
        filtered_dids = [did for did in cum_sampled_dids if datamanager2.get_rel_label(did) == 1]
    else:
        raise NotImplementedError

    if filtered_dids != []:
        features = ranker2.get_feature_by_did(filtered_dids)
        scores = ranker.predict(features)
        threshold = max_or_min_func(scores)
    else:
        threshold = -1

    # rank complete dids
    train_dids1, train_labels1 = datamanager.get_training_data2(z)
    train_dids2, train_labels2 = datamanager2.get_training_data(temp_doc_num)
    train_labels = train_labels1 + train_labels2
    train_features1 = ranker.get_feature_by_did(train_dids1)
    #print(train_features1)
    #print(len(train_labels))
    train_features2 = ranker2.get_feature_by_did(train_dids2)
    train_features = vstack([train_features1,train_features2])
    #print(train_features)
    ranker.train(train_features, train_labels)
    complete_features = ranker2.get_features_by_name('test_dids')
    complete_scores = ranker.predict(test_features)
    #print(scores)
    zipped = sorted(zip(test_dids, complete_scores), key=itemgetter(1), reverse=True)

    # shown dids
    shown_dids = []
    check_func = datamanager2.assess_state_check_func()
    for i, (did, score) in enumerate(zipped):
        if score >= threshold or check_func(did) is True:
            shown_dids.append(did)

    tar_run_file = name_tar_run_file(data_name=data_test, model_name=model_name, topic_set=topic_set,
                                     exp_id=random_state, topic_id=topic_id)
    with open(tar_run_file, 'w', encoding='utf8') as f:
        write_tar_run_file(f=f, topic_id=topic_id, check_func=check_func, shown_dids=shown_dids)

    LOGGER.info('TAR is finished.')

    return

def main(sub_percentage, bound_bt, ita, topic,data_train,data_test,train_percentage):
    data_name = data_train
    topic_id = topic
    topic_set = data_train + '(train' + str(train_percentage) + ')' + data_test + '(test)'
    query_file = os.path.join(PARENT_DIR, 'data', data_name, 'topics', topic_id)
    qrel_file = os.path.join(PARENT_DIR, 'data', data_name, 'qrels', topic_id)
    doc_id_file = os.path.join(PARENT_DIR, 'data', data_name, 'docids', topic_id)
    doc_text_file = os.path.join(PARENT_DIR, 'data', data_name, 'doctexts', topic_id)

    scal_method(data_name, topic_id, topic_set, query_file, qrel_file, doc_id_file, doc_text_file,sub_percentage,bound_bt,ita,data_test,train_percentage)

#main(1.0,110,1.5,'1','android','ceylon',0.1)