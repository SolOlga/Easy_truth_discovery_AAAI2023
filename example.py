import pandas as pd
import numpy as np

from algorithms import PairwiseDistanceFunctions, ProxyDistanceFunctions
from algorithms import TruthDiscoveryAlgorithms
from utils import VotingRules



######### Example 1 #########
# * emotions Continuous data
# * proxy distance function: square euclidian
# * voting rule : mean
# Reading synthetic dataset which its proto population distribution is Normal with mean .65 and 0.15 STD
n = 9
k = 20
dataset = pd.read_csv("data/synthetic/CAT2_k2000_n2000_N(0.65,0.15).csv", header=None).iloc[:n, :k]

print(dataset)
# Reading the appropriate ground truth dataset
dataset_gt = pd.read_csv("data/synthetic/CAT2_k2000_n2000_N(0.65,0.15)_GT.csv", header=None).iloc[:,:k].to_numpy()[0]

print(dataset_gt)
# Calling IPTD for aggregating the dataset
algorithm = TruthDiscoveryAlgorithms.get_algorithm("IPTD")
simple_majority = TruthDiscoveryAlgorithms.get_algorithm("UA")
pairwise_distance = PairwiseDistanceFunctions.get_function("hamming_binary")
simple_majority = TruthDiscoveryAlgorithms.get_algorithm("UA")

answers = algorithm(iterations=10,
                    data=dataset,
                    gt=dataset_gt,
                    voting_rule="mean",
                    iter_weight_method= "1",
                    variant= "PD",
                    mode="continuous",
                    pairwise_distance_function = pairwise_distance,
                    )

MAJ_answers = simple_majority(
                    data=dataset,
                    gt=dataset_gt,
                    voting_rule=None,
                    mode="categorical",
                    params = None
                    )


# Calling the relevant distance function to measure the distance from the ground truth
distance_function = ProxyDistanceFunctions.get_function("square_euclidean")
error = distance_function(answers, dataset_gt)
print(error)
#############################

# # Reading synthetic dataset which its proto population distribution is Normal with mean .65 and 0.15 STD
# dataset = pd.read_csv("data/synthetic/CAT2_k2000_n2000_N(0.65,0.15).csv", header=None).iloc[:20, :20]

# # Reading the appropriate ground truth dataset
# dataset_gt = pd.read_csv("data/synthetic/CAT2_k2000_n2000_N(0.65,0.15)_GT.csv", header=None).iloc[:, :20]

# data_params ={'df': dataset}
# params = {'kernel':"Gaussian"}
# # Calling IPTD for aggregating the dataset
# algorithm = TruthDiscoveryAlgorithms.get_algorithm("KDEm_alg")

# # Lambda = 0 - max likelihood estimator
# #  or 4 - simple variants
# answers = algorithm(data_params=data_params,
#                     params = params
#                     )

# # Calling the relevant distance function to measure the distance from the ground truth
# distance_function = ProxyDistanceFunctions.get_function("hamming_binary")
# error = distance_function(answers, dataset_gt)
# print(error)



# # Reading synthetic dataset which its proto population distribution is Normal with mean .65 and 0.15 STD
# dataset = pd.read_csv("data/synthetic/CAT2_k2000_n2000_N(0.65,0.15).csv", header=None).iloc[:20, :20]

# # Reading the appropriate ground truth dataset
# dataset_gt = pd.read_csv("data/synthetic/CAT2_k2000_n2000_N(0.65,0.15)_GT.csv", header=None).iloc[:, :20]

# # Calling IPTD for aggregating the dataset
# algorithm = TruthDiscoveryAlgorithms.get_algorithm("EVTD")

# # Lambda = 0 - max likelihood estimator
# #  or 4 - simple variants
# answers = algorithm(data=dataset,
#                     gt=dataset_gt
#                     )

# # Calling the relevant distance function to measure the distance from the ground truth
# distance_function = ProxyDistanceFunctions.get_function("hamming_binary")
# error = distance_function(answers, dataset_gt)
# print(error)



# # Reading synthetic dataset which its proto population distribution is Normal with mean .65 and 0.15 STD
# dataset = pd.read_csv("data/Rankings/Dots and Puzzles/Dots-3.csv", header=None).iloc[:20, :20]

# # Reading the appropriate ground truth dataset
# dataset_gt = pd.read_csv("data/Rankings/Dots and Puzzles/plain_4_GT.csv", header=None).iloc[:, :20]

# # Calling IPTD for aggregating the dataset
# algorithm = TruthDiscoveryAlgorithms.get_algorithm("IPTD")
# voting_rule = "vr_plurality"
# pairwise_distance = PairwiseDistanceFunctions.get_function("kendall_tau_distance")

# # Lambda = 0 - max likelihood estimator
# #  or 4 - simple variants
# answers = algorithm(iterations=1,
#                     data=dataset,
#                     gt=dataset_gt,
#                     voting_rule=voting_rule,
#                     variant= "PD",
#                     mode="rankings",
#                     pairwise_distance_function = pairwise_distance,
#                     iter_weight_method= "1"
#                     )

# # Calling the relevant distance function to measure the distance from the ground truth
# distance_function = ProxyDistanceFunctions.get_function("kendall_tau_distance")
# error = distance_function(answers, dataset_gt)
# print(error)