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
import random,sys
# import statistics.median

from .estimator import Estimator
from .utils import run_test
from ..constants import MODEL_ROOT, NUM_THREADS, PKL_PROTO
from ..dtypes import is_categorical
from ..dataset.dataset import load_table
from ..workload.workload import query_2_triple

L = logging.getLogger(__name__)
HL = 31
MOD = 2147483647

class bucket(object):
    """docstring for bucket"""
    def __init__(self,lb,lbv,ub,ubv,_freq,_distinct):
        super(bucket, self).__init__()
        self._lb = (lb,lbv)
        self._ub = (ub,ubv)
        self._freq = _freq
        self._distinct = _distinct
        

class Histogram_Sketch:
    def __init__(self, val_map_all, buckets_all_hist,buckets_all_sketch):
        self.val_map_all = val_map_all
        self.buckets_all_hist = buckets_all_hist
        self.buckets_all_sketch = buckets_all_sketch

class Sketch(object):
    """docstring for Sketch"""
    def __init__(self, _a_vals,_b_vals, _hfunc1,_hfunc2,_hfunc3,_hfunc4,_hfunc5):
        super(Sketch, self).__init__()
        self._a_vals = _a_vals
        self._b_vals = _b_vals
        self._hfunc1 = _hfunc1
        self._hfunc2 = _hfunc2
        self._hfunc3 = _hfunc3
        self._hfunc4 = _hfunc4
        self._hfunc5 = _hfunc5
 
def Random_Generate():
    x = random.randint(0,sys.maxsize)
    h = random.randint(0,sys.maxsize)

    return x ^ ((h & 1) << 31)


def hash31(a,b,x):
  result = (a * x) + b;
  result = ((result >> HL) + result) & MOD;

  return result;

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros      

def count_frequency(_data_column):
    frequency = dict()
    for i in _data_column:
        if i not in frequency:
            frequency[i] = 1
        else:
            frequency[i] += 1

    return frequency


def construct_hist(_table, _num_bins):
    
    columns,val_map_all,buckets_all_hist,buckets_all_sketch = dict(),dict(),dict(),dict()   
    start_stmp = time.time()
    col_type_map = dict()
    check_flag,check_val = False,None
    for i in range(len(_table.data.columns)):
        col_name =_table.data.columns[i]
        col_type = str(_table.columns[col_name].dtype)
        col_type_map[col_name] = col_type

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
        
        total_tuple = len(columns[col_name])
        bucket_size = int(total_tuple/100) # 100 - Numer of bins for histogram
        all_column_buckets = dict()
        buckets = dict()
        cbi = 0 # Current frequency
        prev_key = None


        ########## Construct the Histogram  #################################
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

        buckets_all_hist[col_name] = buckets



        ########## Construct the Sketch  #################################
        a_vals = []
        for i in range(0,5):
            a_vals.append(Random_Generate())
        
        b_vals = []
        for i in range(0,5):
            b_vals.append(Random_Generate())

        hfunc1 = zerolistmaker(_num_bins)
        hfunc2 = zerolistmaker(_num_bins)
        hfunc3 = zerolistmaker(_num_bins)
        hfunc4 = zerolistmaker(_num_bins)
        hfunc5 = zerolistmaker(_num_bins)


        for key,val in col_freq2.items():
            idx1 = hash31(a_vals[0],b_vals[0],key)%_num_bins
            idx2 = hash31(a_vals[1],b_vals[1],key)%_num_bins
            idx3 = hash31(a_vals[2],b_vals[2],key)%_num_bins
            idx4 = hash31(a_vals[3],b_vals[3],key)%_num_bins
            idx5 = hash31(a_vals[4],b_vals[4],key)%_num_bins
            
            hfunc1[idx1] += val*(-1 if Random_Generate()%2==1 else 1)
            hfunc2[idx2] += val*(-1 if Random_Generate()%2==1 else 1)
            hfunc3[idx3] += val*(-1 if Random_Generate()%2==1 else 1)
            hfunc4[idx4] += val*(-1 if Random_Generate()%2==1 else 1)
            hfunc5[idx5] += val*(-1 if Random_Generate()%2==1 else 1)

        sketch = Sketch(a_vals,b_vals,hfunc1,hfunc2,hfunc3,hfunc4,hfunc5)
        buckets_all_sketch[col_name] = sketch

        
    
    hist = Histogram_Sketch(val_map_all,buckets_all_hist,buckets_all_sketch)        
    

    ## Converting the data-structure
    z = 1
    for key,val in hist.buckets_all_hist.items():
        lb = 1
        buckets = []
        for k,v in val.items():
            ub = k
            freq = v
            distinct = (ub-lb+1)
            lbv = [k for k,v in hist.val_map_all[key].items() if v == lb][0]
            ubv = [k for k,v in hist.val_map_all[key].items() if v == ub][0]
            buc = bucket(lb,lbv,ub,ubv,freq,distinct)
            buckets.append(buc)
            lb = k+1
            z += 1
        hist.buckets_all_hist[key] = buckets
    # print("ZZZZZZZZZZZZZZZ # ",z)
    # print("####################### Histogram ###################################################################")
    # ########## Construct the Sketch  #################################
    # z = 1
    # for key,val in hist.buckets_all_hist.items():
    #     print(key)
    #     for buc in val:
    #         # print(buc)
    #         z += 1
    #         print("["+str(buc._lb)+"-"+str(buc._ub)+"] frequency "+str(buc._freq)+" distinct "+str(buc._distinct))
    #     print("\n")
    # print("ZZZZZZZZZZZZZZZ # ",z)
    # print("#####################################################################################################")

    # print("########################## Sketch ###################################################################")
    # for key,val in hist.buckets_all_sketch.items():
    #     print(key)
    #     print("hfunc1 ",val._hfunc1)
    #     print("hfunc1 ",val._hfunc2)
    #     print("hfunc1 ",val._hfunc3)
    #     print("hfunc1 ",val._hfunc4)
    #     print("hfunc1 ",val._hfunc5)
        
    # print("#####################################################################################################")

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
        super(EDHist, self).__init__(table=table, bins=len(partitions.buckets_all_hist))
        self.val_map_all = partitions.val_map_all
        self.buckets_all_hist = partitions.buckets_all_hist
        self.buckets_all_sketch = partitions.buckets_all_sketch
        self.table = table

    def normalize(self,_min,_max,_val,_distinct):
        if _min == _max:
            return 1
        n_val = (_val-_min)/(_max-_min)
        return int(round(n_val*_distinct))
        
    
    def get_points_on_left(self,col,_val,eq):

        card = 0
        for b in self.buckets_all_hist[col]:
            b_ub = b._ub[1]
            b_lb = b._lb[1]
            if b_ub  >= _val:
                if eq == True:
                    card += (b._freq/b._distinct)*self.normalize(b_lb,b_ub,_val+1,b._distinct)
                else:
                    card += (b._freq/b._distinct)*self.normalize(b_lb,b_ub,_val,b._distinct)
                return card
            else:
                card += b._freq

    def calulate_sketch(self,a_vals,b_vals,col_name,key):
        num_bins = len(self.buckets_all_sketch[col_name]._hfunc1)
        idx1 = hash31(a_vals[0],b_vals[0],key)%num_bins
        idx2 = hash31(a_vals[1],b_vals[1],key)%num_bins
        idx3 = hash31(a_vals[2],b_vals[2],key)%num_bins
        idx4 = hash31(a_vals[3],b_vals[3],key)%num_bins
        idx5 = hash31(a_vals[4],b_vals[4],key)%num_bins

        val1 = self.buckets_all_sketch[col_name]._hfunc1[idx1]
        val2 = self.buckets_all_sketch[col_name]._hfunc2[idx2]
        val3 = self.buckets_all_sketch[col_name]._hfunc3[idx3]
        val4 = self.buckets_all_sketch[col_name]._hfunc4[idx4]
        val5 = self.buckets_all_sketch[col_name]._hfunc5[idx5]
    
        card = np.median([val1,val2,val3,val4,val5])
        return card
    # cnt = 0
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
            c_coveredl = self.get_points_on_left(col,l,False)
            c_covered = c_coveredu - c_coveredl
        elif op == '=':
            val = None
            if col_type == 'category':
                val = self.val_map_all[col][valUM]
            else:
                val = valUM
            a_vals = self.buckets_all_sketch[col]._a_vals
            b_vals = self.buckets_all_sketch[col]._b_vals
            col_name = col
            key = val
            c_covered = self.calulate_sketch(a_vals,b_vals,col_name,key)
        
        return c_covered
    
    def query(self, query):
        start_stmp = time.time()
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        est_card = 1

        for i, predicate in enumerate(zip(columns, operators, values)):
            cname, op, val = predicate
            col_type = self.table.columns[cname].dtype # Rounding to nearest integer value
            if op == "=":
                print(predicate)
        # cname = "marital_status"
        # op = "="
        # val = "Married-civ-spouse"
        # col_type = "category"

            if col_type == 'int64' and type(val) != tuple:
                val = int(round(val))
            
            # print("Query #### ",cname,op,val,col_type)
            estC = abs(self.estmtor(cname,op,val,col_type))
            est_S = estC/self.table.row_num # Selectivity
            est_card *= est_S
     
        dur_ms = (time.time() - start_stmp) * 1e3
        return np.round(est_card*self.table.row_num), dur_ms


def load_mhist(dataset: str, model_name: str) -> Tuple[Estimator, Dict[str, Any]]:
    model_file = MODEL_ROOT / dataset / f"{model_name}.pkl"
    L.info(f"load model from {model_file} ...")
    with open(model_file, 'rb') as f:
        state = pickle.load(f)

    table = load_table(dataset, state['version'])
    partitions = state['partitions']
    estimator = MHist(partitions, table)
    return estimator, state

def test_sketches(seed: int, dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        version: the version of table that the histogram is built from, might not be the same with the one we test on
        num_bins: maximum number of partitions
    """ 
    # prioriy: params['version'] (draw sample from another dataset) > version (draw and test on the same dataset)
    print("#############CHECK -- WE ARE HERE##############")
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
    estimator = EDHist(partitions, table)
    L.info(f"Built Histogram Sketch estimator: {estimator}")

    # cnt = 0
    run_test(dataset, version, workload, estimator, overwrite)
    # print("Total point Queries ",cnt)
