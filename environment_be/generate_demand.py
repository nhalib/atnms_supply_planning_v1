# here i generate demand profiles for each customer; almost purely random

from req_py_libraries import *


class _demand_profile():

    def __init__(self,customers,products):

        self.M = customers # LIST of customers
        self.products = products # LIST of products

        self.ts_demand_profile_loc = "_data/demand_profile.xlsx"
        self.ts_demand_profile = []


    # using ideal customer average demand, generate a poisson r.v of demand. then use service level performance to scale it down.
    def _customer_demand_reactive(self, service_level_performance, ideal_customer_avg_demand,cur_epoch):

        # service_level_performance is an exponentially weighted average of maintained service_level
        _reactive_customer_demand = {}

        for customer in self.M:
            for product in self.products:
                t_dict = {}

                t_dict["product"] = product
                t_dict["customer"] = customer
                t_dict["cur_epoch"] = cur_epoch

                dist_lambda = ideal_customer_avg_demand[customer] * 1
                ideal_demand = np.random.poisson(lam=dist_lambda) # ideal demand follows from the [Poisson Distribution]

                # alpha * previously maintained service level + (1-alpha) * current service level
                sl_performance = service_level_performance[customer]

                t_dict["sl_performance"] = sl_performance

                # higher is deviation of service level from 1, sharp increase in the probability of a demand less than the ideal amount
                target_demand = ideal_demand * np.random.uniform(sl_performance*0.6,sl_performance*0.7,1)[0]

                if np.random.uniform(0,1,1) > 0.95:
                    target_demand = target_demand**2 # suddenly a very high demand

                _reactive_customer_demand[(product,customer)] = max(int(target_demand),1)
                #print(product,customer,_reactive_customer_demand[(product,customer)])

                self.ts_demand_profile.append(t_dict)

        return [_reactive_customer_demand]


    def _save_outputs(self):

        ts_demand_profile_df = pd.DataFrame(self.ts_demand_profile)
        ts_demand_profile_df.to_excel(self.ts_demand_profile_loc)







