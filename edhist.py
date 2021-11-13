import enum
import copy
import logging
import pickle
import time
import threading
import bisect
from typing import Any, Dict, Tuple
import numpy as np
from collections import Counter 
import math

from .estimator import Estimator
from .utils import run_test
from ..constants import MODEL_ROOT, NUM_THREADS, PKL_PROTO
from ..dtypes import is_categorical
from ..dataset.dataset import load_table
from ..workload.workload import query_2_triple

L = logging.getLogger(__name__)


class bucket(object):
    """docstring for bucket"""
    def __init__(self,lb,lbv,ub,ubv,_freq,_distinct):
        super(bucket, self).__init__()
        self._lb = (lb,lbv)
        self._ub = (ub,ubv)
        self._freq = _freq
        self._distinct = _distinct
        

class Histogram:
    def __init__(self, val_map_all, buckets_all):
        self.val_map_all = val_map_all
        self.buckets_all = buckets_all

def count_frequency(_data_column):
    frequency = dict()
    for i in _data_column:
        if i not in frequency:
            frequency[i] = 1
        else:
            frequency[i] += 1

    return frequency


def construct_hist(_table, _num_bins):
    
    columns,val_map_all,buckets_all = dict(),dict(),dict()
    start_stmp = time.time()
    for i in range(len(_table.data.columns)):
        col_name =_table.data.columns[i]
        columns[col_name] = sorted(list(_table.data[col_name]))
        
        col_freq1 = count_frequency(columns[col_name]) # Counting frequency and most frequent values [top 3 and if freq > bucket size]        

        # Map columns_frequency dictionay keys to sequencial int values
        val_map = dict()
        col_freq2 = dict()
        i = 1
        for key,val in col_freq1.items():
            val_map[key] = i
            col_freq2[i] = val
            i += 1

        val_map_all[col_name] = val_map
        
        # Building histogram for every column
        total_tuple = len(columns[col_name])
        bucket_size = int(total_tuple/_num_bins)
        all_column_buckets = dict()
        buckets = dict()
        cbi = 0 # Current frequency
        prev_key = None


        # Constructing the buckets
        i = 0
        CS = 0
        for key,val in col_freq2.items():
            i += 1
            if val >= bucket_size : # Maximum frequency item
                if prev_key != None:
                    
                    if prev_key not in buckets:
                        buckets[prev_key] = cbi
                    buckets[key] = val
                    cbi = 0
                    prev_key = key        
                    CS += val                    
                    continue
                else:
                    buckets[key] = val
                    prev_key = key
                    CS += val                    
                    continue

            elif cbi + val > bucket_size :
                l = bucket_size - cbi
                r = cbi + val - bucket_size
                if l > r:
                    buckets[key] = cbi+val
                    cbi = 0
                    prev_key = key
                    CS += val                    
                    continue
                else:
                    buckets[prev_key] = cbi
                    cbi = 0
                    CS += val
                    
            
            if i == len(col_freq2) and key not in buckets:
                buckets[key] = cbi+val
                CS += val

            cbi = cbi + val
            prev_key = key

        
        buckets_all[col_name] = buckets
    
    hist = Histogram(val_map_all,buckets_all)        
    # print(hist.val_map_all,"\n\n")
    # print("#######################Val Map###################################")

    # for key,val in hist.val_map_all.items():
    #     print(key,val)
    #     print("SUM VAL MAP ",key,sum(val.values()))
    # print("#######################buckets###################################")
    # print("Size of buckets ",total_tuple/10,"\n")
    # for key,val in hist.buckets_all.items():
    #     print(key,val)
    #     # print("SUM buckets ",key,sum(val.values()))
    # print("##########################################################")


    ## Converting the data-structure
    for key,val in hist.buckets_all.items():
        lb = 1
        buckets = []
        for k,v in val.items():
            ub = k
            freq = v
            distinct = (ub-lb+1)
            lbv = [k for k,v in hist.val_map_all[key].items() if v == lb][0]
            ubv = [k for k,v in hist.val_map_all[key].items() if v == ub][0]
            # print(type(lbv),"----- type")
            buc = bucket(lb,lbv,ub,ubv,freq,distinct)
            buckets.append(buc)
            lb = k+1
        hist.buckets_all[key] = buckets

    print("####################### After Conversion ###################################")
    for key,val in hist.buckets_all.items():
        print(key)
        for buc in val:
            print("["+str(buc._lb)+"-"+str(buc._ub)+"] frequency "+str(buc._freq)+" distinct "+str(buc._distinct))
        print("\n")
    print("##########################################################")


    dur_min = (time.time() - start_stmp) / 60
    print("####### Time Seconds ############",dur_min*60)
    # L.info(f'Construct Hist (EDHIST) finished, use {len(buc)} partitions ({hist_size:.2f}MB)! Time spent since start: {dur_min:.2f} mins')

    # # print("Partitions # ",partitions)
    state = {
        'device': 'cpu',
        'threads': NUM_THREADS,
        'dataset': _table.dataset,
        'version': _table.version,
        'partitions': hist,
        'train_time': dur_min,
        'model_size': len(buckets),
    }

    print(state['partitions'])
    return state


class EDHist(Estimator):
    def __init__(self, partitions, table):
        super(EDHist, self).__init__(table=table, bins=len(partitions.buckets_all))
        self.val_map_all = partitions.val_map_all
        self.buckets_all = partitions.buckets_all
        self.table = table

    def normalize(self,_min,_max,_val,_distinct):
        if _min == _max:
            return 1
        n_val = (_val-_min)/(_max-_min)
        return int(round(n_val*_distinct))
        
    
    def get_points_on_left(self,col,_val,eq):

        card = 0
        for b in self.buckets_all[col]:
            b_ub = b._ub[1]
            b_lb = b._lb[1]
            # print(" <<< u l v card",b_ub,b_lb,_val,card)
            if b_ub  >= _val:
                if eq == True:
                    #card += (b._freq/b._distinct)*(_val-b_lb+1)
                    card += (b._freq/b._distinct)*self.normalize(b_lb,b_ub,_val+1,b._distinct)
                else:
                    # card += (b._freq/b._distinct)*(_val-b_lb)
                    card += (b._freq/b._distinct)*self.normalize(b_lb,b_ub,_val,b._distinct)
                    # print("CHECK ",(b._freq/b._distinct),(_val-b_lb),card)
                return card
            else:
                card += b._freq

    def estmtor(self,col,op,valUM,col_type):
        domain_size = len(self.table.data[col])
        c_covered = 0

        if op == '<':
            c_covered = self.get_points_on_left(col,valUM,False)
        elif op == '<=':
            c_covered = self.get_points_on_left(col,valUM,True)
        elif op == '>':
            c_covered = domain_size - self.get_points_on_left(col,valUM,True)
        elif op == '>=':
            c_covered = domain_size - self.get_points_on_left(col,valUM,False)
        elif op == '[]':
            l,u = int(round(valUM[0])),int(round(valUM[1]))
            c_coveredu = self.get_points_on_left(col,u,True)
            # print("u estu ",u,c_coveredu) 
            c_coveredl = self.get_points_on_left(col,l,False)
            # print("l estl ",l,c_coveredl)
            c_covered = c_coveredu - c_coveredl
        elif op == '=':
            if col_type == 'category':
                val = self.val_map_all[col][valUM]
                for b in self.buckets_all[col]:
                    # print("Here 1 ",val,b._ub[0],type(val),type(b._ub[0]))
                    if b._ub[0] >= val:
                        c_covered = (b._freq/b._distinct)
                        # print("Here 2 ",c_covered,b._freq,b._distinct)
                        return c_covered
            else:
                for b in self.buckets_all[col]:
                    b_ub = b._ub[1]
                    b_lb = b._lb[1]
                    # print(" ==== u l v ",b_ub,b_lb,valUM)
                    if b_ub  >= valUM:
                        c_covered = (b._freq/b._distinct)
                        return c_covered
        
        # print("Estimator ### ",col,op,valUM,c_covered)
        return c_covered
    
    def query(self, query):
        start_stmp = time.time()
        # print(query)
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        est_card = 1


        # descritize predicate parameters for non-numerical columns
        for i, predicate in enumerate(zip(columns, operators, values)):
            cname, op, val = predicate
            # print(self.table.columns[cname],type(self.table.columns[cname]))
            col_type = self.table.columns[cname].dtype # Rounding to nearest integer value
            if col_type == 'int64' and type(val) != tuple:
                val = int(round(val))
            
            estC = self.estmtor(cname,op,val,col_type)
            # print("col type ",col_type,cname,op,val," card ",estC)
            est_S = estC/self.table.row_num # Selectivity

            est_card *= est_S
     
        dur_ms = (time.time() - start_stmp) * 1e3

        # #  return np.round(est.card.sum()), dur_ms
        return np.round(est_card*self.table.row_num), dur_ms


def load_mhist(dataset: str, model_name: str) -> Tuple[Estimator, Dict[str, Any]]:
    model_file = MODEL_ROOT / dataset / f"{model_name}.pkl"
    L.info(f"load model from {model_file} ...")
    with open(model_file, 'rb') as f:
        state = pickle.load(f)

    table = load_table(dataset, state['version'])
    partitions = state['partitions']
    #  print_partitions(partitions)
    estimator = MHist(partitions, table)
    return estimator, state

def test_edhist(seed: int, dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        version: the version of table that the histogram is built from, might not be the same with the one we test on
        num_bins: maximum number of partitions
    """
    # prioriy: params['version'] (draw sample from another dataset) > version (draw and test on the same dataset)
    print("#############CHECK##############")
    table = load_table(dataset, params.get('version') or version)

    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}-edhist_bin{params['num_bins']}.pkl"

    if model_file.is_file():
        L.info(f"{model_file} already exists, directly load and use")
        with open(model_file, 'rb') as f:
            state = pickle.load(f)
    else:
        L.info(f"Construct EDHist with at most {params['num_bins']} bins...")
        state = construct_hist(table, params['num_bins'])
        with open(model_file, 'wb') as f:
            pickle.dump(state, f, protocol=PKL_PROTO)
        L.info(f"EDHist saved to {model_file}")

    partitions = state['partitions']
    # print(partitions)
    estimator = EDHist(partitions, table)
    # L.info(f"Built MHist estimator: {estimator}")

    run_test(dataset, version, workload, estimator, overwrite)
