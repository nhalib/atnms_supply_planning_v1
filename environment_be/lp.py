from req_py_libraries import *
from environment_be.product_mapping import _product_mapping

class _readjust_lp():

    def __init__(self, demand_profile, components_profile, customer_profile, action):
        p1 = _product_mapping()
        [self.map_df] = p1._build_map_df()

        self.demand_profile = demand_profile # a dictionary: product,customer => demand
        self.components_profile = components_profile # a dictionary: component => amount available
        self.customer_profile = customer_profile # a dictionary: customer => rank
        self.type = action # this is the action taken by the RL agent


    # non uniform weight allocation
    def _nonuniform_weights(self,scale_factor,customers):

        norm_constant = 0
        for customer in customers:
            temp_val = np.exp(-non_uniform_weight_factor * self.customer_profile[customer])
            scale_factor[customer] = temp_val
            norm_constant += temp_val

        for customer in customers:
            scale_factor[customer] = scale_factor[customer]

        return [scale_factor]


    # uniform weight allocation
    def _uniform_weights(self,scale_factor,customers):

        for customer in customers:scale_factor[customer] = 1/len(customers)

        return [scale_factor]

    def _run_lp(self):
        model = ConcreteModel()

        # amount of each component to environment_be built
        components = self.map_df.columns.tolist()
        products = self.map_df.index.tolist()
        customers = list(self.customer_profile.keys())

        scale_factor = {}
        for customer in customers:scale_factor[customer] = 1 # assign equal importance to all customers

        model.U = Var(products,customers,domain=NonNegativeReals) # fill rate of product i for customer k
        model.product_components = Var(products,customers,components, domain=NonNegativeReals) # amount of component j allocated to product i for customer k
        model.products = Var(products,customers,domain=NonNegativeReals) # amount of product i allocated to customer k

        model.compositions = ConstraintList()
        model.conservations = ConstraintList()
        model.obj_handler = ConstraintList()


        # product - customer to product - customer - component
        for product in self.map_df.index:
            for customer in customers:
                model.compositions.add(
                    model.products[product,customer] == sum(model.product_components[product,customer,component] * self.map_df.loc[product,component] for component in components)
                )


        # conservation of total amount of component
        for component in components:
            model.conservations.add(
               self.components_profile[component] ==
               sum(model.product_components[product,customer,component] for product in products for customer in customers)
            )


        # action [0,2] scale factor
        if self.type in [0]:
            [scale_factor] = self._uniform_weights(scale_factor=scale_factor, customers=customers)

        # action [1,3] scale factor
        elif self.type in [1]:
            [scale_factor] = self._nonuniform_weights(scale_factor=scale_factor,customers=customers)


        #if self.type in [1,2]:
        if True:
             for product in products:
                for customer in customers:
                    model.conservations.add(model.products[product,customer] <= self.demand_profile[product,customer])


        if self.type in [0]:
            # if action is 0, give equal priority to each ,product,customer
            model.service = Objective(expr=sum(model.products[product,customer]/self.demand_profile[product,customer]
                                               for product in products for customer in customers),
                                      sense=maximize)
        elif self.type in [1]:
            # if action is 1, give more priority to customers with larger <product,customer>
            model.service = Objective(expr=sum((model.products[product,customer]* scale_factor[customer])/self.demand_profile[product,customer]
                                               for product in products for customer in customers),
                                      sense=maximize)

        elif self.type in [2]:
            # if action is 2, only prioritise the top customers (ones with priority rank < 3)
            model.service = Objective(expr=sum((model.products[product,customer]* scale_factor[customer])/self.demand_profile[product,customer]
                                               for product in products for customer in customers
                                               if self.customer_profile[customer] < 3),
                                      sense=maximize)


        op = SolverFactory('cbc').solve(model)

        op_dict = {}
        for product in products:
            for customer in customers:
                op_dict[(product,customer)] = model.products[product,customer]()

        return [op.solver.status,op.solver.termination_condition,op_dict]


# demand profile is the state to the system/
def _execute_lp(demand_profile,components_profile,customer_profile,action):

        r1 = _readjust_lp(demand_profile=demand_profile,
                          components_profile=components_profile,
                          customer_profile=customer_profile,action=action)

        [solver_status,termination_condition,new_mix] = r1._run_lp()

        return [solver_status,termination_condition,new_mix]