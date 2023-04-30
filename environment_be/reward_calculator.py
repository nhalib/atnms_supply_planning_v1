from req_py_libraries import *



class reward_calculator():

    def __init__(self, customer_rank,products):

        self.customer_reeval_tframe = 3

        self.customer_rank = customer_rank
        self.products = products

        self.sum_of_customer_ranks = len(self.customer_rank.keys()) * (len(self.customer_rank.keys()) + 1)

        self.ts_imm_reward_loc = "_data/immediate_reward.xlsx"
        self.ts_cumul_reward_loc = "_data/cumul_reward.xlsx"

        self.customer_satisfaction = {} # a dictionary: customer => cumulative satisfaction
        for key,value in self.customer_rank.items():
            self.customer_satisfaction[key] = 0

        self.ts_cumul_reward_df = []
        self.ts_immediate_reward_df = []

        self.profit_per_product = 10


    def _service_level_dict_builder(self,customer,tgt_dict,sl):

        if customer in tgt_dict:tgt_dict[customer].append(sl)
        else:tgt_dict[customer] = [sl]

        return [tgt_dict]


    def _get_customer_rank_score(self,customer):

        score = len(self.customer_rank.keys()) - self.customer_rank[customer] # scaled rank, giving numerical superiority to rank 1 vs. rank 7

        return [score]


    def calculate_reward(self, solver_status, termination_condition, new_mix, demand_profile, cur_epoch,service_level_performance):

        cond_a = solver_status == 'ok'
        cond_b = termination_condition == 'optimal'
        action_reward = 0
        if not cond_b:
            action_reward += -10 # super high penalty if the solver has a non-optimal/infeasible/unbounded solution
        else:

            epoch_end_customer_service_level_dict = {}

            # key => (product,customer)
            for key,value in demand_profile.items():
                t_dict = {}

                # for diagnosis
                t_dict["epoch"] = cur_epoch
                t_dict["product"] = key[0]
                t_dict["customer"] = key[1]

                target_amt = demand_profile[key]
                available_amt = new_mix[key]

                # for diagnosis
                t_dict["demand"] = target_amt
                t_dict["allocated"] = available_amt

                temp_service_level = min(1,available_amt/target_amt)
                unmet_service_level = 1 - temp_service_level

                [epoch_end_customer_service_level_dict] = self._service_level_dict_builder(customer=key[1],tgt_dict=epoch_end_customer_service_level_dict,sl=temp_service_level)

                # for diagnosis
                t_dict["service_level"] = temp_service_level

                customer_rank_score = self._get_customer_rank_score(customer=key[1])[0]

                # for diagnosis
                t_dict["customer_rank_score"] = customer_rank_score

                # immediate reward due to performance of RL agent for (product,customer) in current epoch
                # profit per customer sold. this is the reward to the RL agent
                instance_reward = available_amt * self.profit_per_product

                action_reward += instance_reward

                # for diagnosis
                t_dict["reward"] = instance_reward

                self.customer_satisfaction[key[1]] += instance_reward # adding reward to cumulative customer satisfaction (cumulative across all products)

                self.ts_immediate_reward_df.append(t_dict)


            for customer in self.customer_rank:
                service_level_performance[customer] = 0.6 * service_level_performance[customer] + \
                                                      0.4 * np.mean(epoch_end_customer_service_level_dict[customer])


        # after every 'm' periods, customers re-evaluate the performance of us, the supplier, based on
        # accumulated service level over the customer_(re-eval)_tframe
        if cur_epoch % self.customer_reeval_tframe == 0 and cur_epoch > 0 and False:

            for customer,cum_reward in self.customer_satisfaction.items():

                t_dict = {}

                # come up with a measure to identify how bad the overall performance has been
                cumulative_reward = self.customer_satisfaction[customer]
                customer_rank_score = self._get_customer_rank_score(customer=customer)[0]

                t_dict["customer"] = customer
                t_dict["cumulative_reward"] = cumulative_reward
                t_dict["customer_rank_score"] = customer_rank_score

                # best case scenario is if the customer has service level 1 across all products for the entire time horizon.
                best_case_scenario = (customer_rank_score/self.sum_of_customer_ranks) * (self.customer_reeval_tframe + 1) * len(self.products) #number of products

                t_dict["best_scenario"] = best_case_scenario

                # penalty here is the deviation from best case scenario. higher the deviation and higher the customer rank, higher is the penalty.
                instance_penalty = (customer_rank_score/self.sum_of_customer_ranks) * (abs(abs(cumulative_reward)-best_case_scenario)/best_case_scenario)
                t_dict["deviation_from_best_performance"] = abs(abs(cumulative_reward)-best_case_scenario)/best_case_scenario

                action_reward -= instance_penalty

                t_dict["penalty for cumul. performance"] = instance_penalty

                # reset cumulative reward for next re-evaluating timeframe
                self.customer_satisfaction[customer] = 0

                self.ts_cumul_reward_df.append(t_dict)


        return [action_reward,service_level_performance]



    def _save_outputs(self):

        ts_immediate_reward_df = pd.DataFrame(self.ts_immediate_reward_df)
        ts_immediate_reward_df.to_excel(self.ts_imm_reward_loc)

        ts_cumul_reward_df = pd.DataFrame(self.ts_cumul_reward_df)
        ts_cumul_reward_df.to_excel(self.ts_cumul_reward_loc)