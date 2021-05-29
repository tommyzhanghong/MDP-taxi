# MDP
a toy problem to demostrate Markov Decision Process (MDP)

Supppose Taxi drivers can pick up customers from any location. When they do not have any customers onboard, taxi drivers need to go somewhere to search for customers. Here we aim to solve this problem using MDP

When fined fare from each location to another, probability of pick up customer at each location, and probability of a customer wanting to go from location A to B, this problem can be solved by policy iteration

With defined transition probability and action rules (can be replaced for different problem), the MDP solution is in the file taxi_MDP_model_policy_Iteration.py
