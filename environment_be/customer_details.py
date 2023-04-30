import pandas as pd

from req_py_libraries import *

class customer_details():

    def __init__(self,demand_ul):
        self.customers = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        self.customer_rank = {}
        self.demand_ul = demand_ul

        self.hard_coded_rank = {
            "A":6,
            "B":1,
            "C":0,
            "D":5,
            "E":3,
            "F":4,
            "G":2
        }


    # non uniform weight allocation
    def _nonuniform_weights(self,customer_rank):

       temp_val = np.exp(-non_uniform_weight_factor*0.05 * customer_rank)
       return [temp_val]


    # routine to generate a customer ranking
    # generate average no. of units ordered by customer for a product per epoch.
    def _ideal_customer_behaviours(self):

        customer_rank_df = []
        ideal_customer_demand_df = []

        for customer in self.customers:
            self.customer_rank[customer] = self.hard_coded_rank[customer]
            customer_rank_df.append({customer:self.customer_rank[customer]})

        ideal_customer_max_demand = {}

        for customer in self.customers:
            customer_rank = self.customer_rank[customer]
            max_demand = self.demand_ul * self._nonuniform_weights(customer_rank=customer_rank)[0]
            ideal_customer_max_demand[customer] = max_demand
            ideal_customer_demand_df.append(ideal_customer_max_demand)

        pd.DataFrame(customer_rank_df).to_excel("_data/customer_rank.xlsx",index=False)
        pd.DataFrame(ideal_customer_demand_df.append(ideal_customer_max_demand)).to_excel("_data/ideal_demand.xlsx",index=False)

        return [self.customer_rank, ideal_customer_max_demand]