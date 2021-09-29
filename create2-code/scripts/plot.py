import numpy as np

import os
import math
import decimal
import sys
import matplotlib.pyplot as plt
from pprint import pprint

def number_of_steps_till_episode(steps_per, ep):
    return np.sum(steps_per[0:ep])

path = sys.argv[1]

run_folders = os.listdir(path)

all_ss = np.array([])
all_ret = np.array([])
all_len = np.array([])

run_folders.sort(key=lambda x: int(x[4:]))

print(run_folders)
for run in run_folders:
    try:
        ss = np.load(path + run + '/data/ep_ss.npy')
        rets = np.load(path + run + '/data/ep_rets.npy')
        lens = np.load(path + run + '/data/ep_lens.npy')
        all_ss = np.concatenate([all_ss, ss])
        all_ret = np.concatenate([all_ret, rets])
        all_len = np.concatenate([all_len, lens])
    except:
        print(run, " not found")

print(all_ss)
print(all_ret)
print(all_len)
print("NUM_EPS", len(all_ss))

unique_ss, counts_ss = np.unique(all_ss, return_counts=True)
print('Unique states ', unique_ss)
print('Frequency of states ', counts_ss)

assert(len(all_ss) == len(all_ret) and len(all_ret) == len(all_len))

state_dict = dict()

for index, s in enumerate(all_ss):
    if s not in state_dict:
        state_dict[s] = [[], [], []]
    state_dict[s][0].append(all_ret[index])
    state_dict[s][1].append(all_len[index])
    state_dict[s][2].append(index)

ordering = []
for s, data in state_dict.items():
    print("STATE ", s, np.mean(data[0]))
    smoothed = data[0]
    smoothing_factor = 0
    for idex in range(smoothing_factor,len(smoothed)):
        smoothed[idex] = (np.mean(smoothed[idex - smoothing_factor:idex+1]))
    plt.plot(data[2], smoothed, label='state '+str(int(s)))
    ordering.append((s, np.mean(state_dict[s][0]), np.std(state_dict[s][0])))
ordering.sort(key=lambda x: x[1])
pprint(ordering)
for stage in range(1, len(ordering) + 1):
    print('slice', np.array(ordering)[0:stage,1:2])
    sliced = -1 * np.array(ordering)[0:stage,1:2] / 107.37 + 16.09871
    print(sliced/np.sum(sliced))

print(all_len)
print('hi', number_of_steps_till_episode(all_len, 200))
print(np.mean(all_ret))
print(len(all_ret))

"""
print(ordering)
tots_m_order = np.full(len(ordering), np.sum(ordering)) - ordering
print(tots_m_order/np.sum(tots_m_order))
"""
"""
ereturns = []
for i in range(len(ordering)):
    ereturns.append(decimal.Decimal(math.e) ** decimal.Decimal(ordering[i][1]))

sum_e = decimal.Decimal(0.0)
for exi in ereturns:
    sum_e += exi

# calculate softmax
for exi in ereturns:
    print(exi/sum_e)

# calculate archit schema
sume_exi = []
tot_sume_exi = decimal.Decimal(0.0)
for exi in ereturns:
    sume_exi.append(sum_e - exi)
    tot_sume_exi += (sum_e - exi)

probs = []
for exi in ereturns:
    probs.append( (sum_e - exi) / tot_sume_exi)
    #     probs.append(tot_sume_exi - exi
print(probs)
"""
plt.legend()
plt.show()
