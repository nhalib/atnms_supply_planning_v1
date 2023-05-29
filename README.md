# atnms_supply_planning_v1
RL agent to perform autonomous supply planning

The full description of this project is on: https://hackernoon.com/pitfalls-in-ai-based-learning-a-supply-chain-example

Key aspects:

a) Environment emulating a supply chain where customers have stochastic demand for goods. Customers have a penalty imposed on us (supply chain) if their demand isn't met by us.
b) Planning of resources and allocation learnt by (deep) Q-learning, with a simple feed-forward neural network to approximate the state-action value function.
