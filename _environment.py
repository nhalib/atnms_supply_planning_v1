# model environment the RL agent is interacting with
# rank of customers is built into the environment

from req_py_libraries import *
from environment_be.lp import _execute_lp
from environment_be.generate_demand import  _demand_profile
from environment_be.product_mapping import _product_mapping
from environment_be.reward_calculator import reward_calculator
from environment_be.customer_details import  customer_details

class _environment():

    def __init__(self):

        p1 = _product_mapping()
        [components_profile] = p1._build_components_profile(qty=500)

        products = p1.products

        c1 = customer_details(demand_ul=100)
        [customer_rank, ideal_customer_avg_demand] = c1._ideal_customer_behaviours()

        self.customer_rank = customer_rank # a dictionary: customer => rank
        self.products = products
        self.components_profile = components_profile

        self.cur_epoch = 0
        self.regime_change_interval = 500 # after every 60 epochs, the regime might change

        self.r1 = reward_calculator(customer_rank=self.customer_rank,products=self.products)

        # initialising service level performance
        self.service_level_performance = {} # a continually evaluated exp. weighted avg. of customer's average service level across products
        for customer in self.customer_rank:self.service_level_performance[customer] = 1

        self.demand_generator = _demand_profile(customers=sorted(self.customer_rank),products=self.products)

        self.ideal_customer_avg_demand = ideal_customer_avg_demand

        self.lt_performance_df = []

        self.regime_change_flag = True

        self._demand_level = 100


    # first call to environment; generate a state to start
    def _environment_reset(self):

        # reset service level performance
        self.service_level_performance = {} # a continually evaluated exp. weighted avg. of customer's average service level across products
        for customer in self.customer_rank:self.service_level_performance[customer] = 1

        # reset ideal customer average demand
        c1 = customer_details(demand_ul=100)
        [_, self.ideal_customer_avg_demand] = c1._ideal_customer_behaviours()

        self.cur_epoch = 0

        # generate a new_state to start of the environment emulation.
        [demand_profile] = self.demand_generator._customer_demand_reactive(
            service_level_performance=self.service_level_performance,
            ideal_customer_avg_demand=self.ideal_customer_avg_demand,cur_epoch=self.cur_epoch)

        return [list(demand_profile.values())]


    # simulate change in overall demand profile, as a regime change
    def _regime_change(self):

        s = np.random.binomial(1,0.5,1)

        if s == 0: tgt_demand_ul = 100 # low demand
        else: tgt_demand_ul = 700 # high demand

        self._demand_level = tgt_demand_ul
        c1 = customer_details(demand_ul=tgt_demand_ul)
        [_, ideal_customer_avg_demand] = c1._ideal_customer_behaviours()

        return [ideal_customer_avg_demand]


    # primary shell to run the environment
    def _daily_interaction(self, cur_action):

        if True:

            # generate_new_state
            [demand_profile] = self.demand_generator._customer_demand_reactive(
                service_level_performance=self.service_level_performance,
                ideal_customer_avg_demand=self.ideal_customer_avg_demand,cur_epoch=self.cur_epoch)

            # LP output is reward and feasibility
            [solver_status, termination_condition, new_mix] = _execute_lp(demand_profile=demand_profile,
                                                                          components_profile=self.components_profile,
                                                                          customer_profile=self.customer_rank,
                                                                          action=cur_action)

            # calculate reward based on the LP output vs demand.
            [state_action_reward,self.service_level_performance] = self.r1.calculate_reward(solver_status=solver_status,
                                                                               termination_condition=termination_condition,
                                                                               new_mix=new_mix,
                                                                               demand_profile=demand_profile,
                                                                               cur_epoch=self.cur_epoch,
                                                                               service_level_performance=self.service_level_performance)

            if self.cur_epoch > 0:
                temp_dict = {"epoch":self.cur_epoch,"demand_level":self._demand_level}
                for customer in self.service_level_performance:temp_dict[customer] = self.service_level_performance[customer]
                self.lt_performance_df.append(temp_dict)


            if self.cur_epoch % 1000 == 0 and self.cur_epoch > 0:
                [self.ideal_customer_avg_demand] = self._regime_change()


            self.cur_epoch += 1

        return [state_action_reward,demand_profile]


    # probes on the environment
    def _environment_lt_perf_data(self):
        return [self.lt_performance_df]


    # call back from RL agent to get next step,reward
    def _environment_step(self,suggested_action):

        [state_action_reward, demand_profile] = self._daily_interaction(cur_action=suggested_action)
        # return reward and demand profile as a vector
        return [state_action_reward, list(demand_profile.values())]



if __name__ == "__main__":
    pass
