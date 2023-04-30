from req_py_libraries import *
from environment_be.generate_demand import _demand_profile
from environment_be.product_mapping import _product_mapping
from environment_be.customer_details import customer_details

from environment_be.lp import _execute_lp

if __name__ == "__main__":

    demand_profile = {
        ('p1','A'):37,
        ('p1', 'B'):33,
        ('p1', 'C'):3100,
        ('p1', 'D'):243,
        ('p1', 'E'):220,
        ('p1', 'F'):220,
        ('p1', 'G'):1985313,
        ('p2', 'A'):121,
        ('p2', 'B'):154,
        ('p2', 'C'):46000,
        ('p2', 'D'):61,
        ('p2', 'E'):54,
        ('p2', 'F'):3395758,
        ('p2', 'G'):4262706,
        ('p3', 'A'):148,
        ('p3', 'B'):92,
        ('p3', 'C'):96,
        ('p3', 'D'):85,
        ('p3', 'E'):12800858,
        ('p3', 'F'):204,
        ('p3', 'G'):236
    }

    components_profile = {
        'c1':1000,
        'c2':1000,
        'c3':1000,
        'c4':1000,
        'c5':1000,
        'c6':1000
    }

    customer_profile = {
        "A":6,
            "B":1,
            "C":0,
            "D":5,
            "E":3,
            "F":4,
            "G":2
    }

    action = 2

    [solver_status,termination_condition,new_mix] = _execute_lp(demand_profile=demand_profile,
                components_profile=components_profile,
                customer_profile=customer_profile,action=action)
    print(new_mix)
    total_reward = 0
    for key in new_mix:
        total_reward += new_mix[key]  * 10
    print(total_reward)