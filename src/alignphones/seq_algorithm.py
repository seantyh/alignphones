import numpy as np
from functools import cache
from panphon import FeatureTable
from typing import List
from typing import Tuple, Dict
TrellisLoc = Tuple[int, int]
Dist = float
BacktrackTable = Dict[TrellisLoc, Tuple[TrellisLoc, Dist]]
AlignPath = List[Tuple[int, int]]

def compute_phone_dist(tgt_phone, ft: FeatureTable, allo_ipas: List[str]):

    @cache
    def inner_func(tgt_phone):
        "a closure is used for the cache mechanism, which needs hashable arguments."
        mask = np.array([x in ft.seg_dict for x in allo_ipas])
        if tgt_phone not in ft.seg_dict:
            dist_vec = np.ones((len(allo_ipas),))*15
        else:
            phone_dist = [ft.fts(tgt_phone).hamming_distance(ft.fts(allo_phone_x)) # type: ignore
                            for allo_phone_x in allo_ipas
                            if allo_phone_x in ft.seg_dict]
            dist_vec = np.zeros((len(allo_ipas),))
            dist_vec[~mask] = np.max(phone_dist)+1
            dist_vec[mask] = phone_dist
        return dist_vec
    return inner_func(tgt_phone)

def compute_scores(prob_mat, tgt_phone, ft: FeatureTable, allo_ipas: List[str]):
    dist_vec = compute_phone_dist(tgt_phone, ft, allo_ipas)  # type: ignore
    dist_vec = np.exp(-dist_vec)    
    dist_vec /= np.sum(dist_vec)
    # logit_mat: M X V
    # dist_vec: V
    score_vec = prob_mat.dot(dist_vec)    
    Z = np.sum(score_vec)
    p_vec = score_vec/Z
    return p_vec

def compute_trellis(prob_mat, 
        epi_phones, 
        ft: FeatureTable, 
        allo_ipas: List[str], 
        C_del=-np.log(0.01)):

    M, V = prob_mat.shape
    N = len(epi_phones)
    trellis = np.zeros((M, N))
    trellis[0,:] = np.arange(N)
    trellis[:,0] = np.arange(M)    
    backtrack: BacktrackTable = {(0,i): ((0,i-1),i) for i in range(1,M)}
    backtrack.update({(j,0): ((j-1,0),j) for j in range(1,N)})
    for j in range(1,N):
        mu_j = -np.log(compute_scores(prob_mat, epi_phones[j], ft, allo_ipas))
        for i in range(1,M):
            dist_vec = [
                # replace
                trellis[i-1, j-1] + mu_j[i],
                # delete (skip epi_phone)
                trellis[i, j-1] + C_del,
                # insert (stay at same epi_phone)
                trellis[i-1, j] + mu_j[i]
            ]
            dist_argmin = np.argmin(dist_vec)
            pointer = ((i-1,j-1),(i,j-1),(i-1,j))[dist_argmin]
            mindist_x = np.min(dist_vec)
            backtrack[(i,j)] = (pointer, mindist_x)        
            trellis[i,j] = mindist_x
    return trellis, backtrack
        