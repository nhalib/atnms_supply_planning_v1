# define a mapping for different components go into a product etc.
# let there environment_be 2 levels. At the base level, you have 6 components being produced. They go into 3 products at different ratios.
# define the ratios here.

from req_py_libraries import *
from environment_be.customer_details import customer_details

class _product_mapping():

    def __init__(self):

        self.components = ['c1','c2','c3','c4','c5','c6']
        self.products = ['p1','p2','p3']
        self.customers = customer_details(demand_ul=10).customers


    def _mapping(self):

        self.map = {
            "p1":{"c1":0.1,
                  "c2":0.3,
                  "c3":0,
                  "c4":0.3,
                  "c5":0.1,
                  "c6":0.2,
                },
            "p2": {"c1":0.3,
                  "c2":0.3,
                  "c3":0.2,
                  "c4":0.1,
                  "c5":0.05,
                  "c6":0.05,
                },
            "p3": {"c1":0.05,
                "c2":0.05,
                "c3":0.1,
                "c4":0.3,
                "c5":0.25,
                "c6":0.25,
                }
        }

    def _build_map_df(self):
        self._mapping()
        return [pd.DataFrame.from_dict(self.map).T]


    def _build_components_profile(self,qty):

        dict = {}
        for component in self.components:
            dict[component] = qty

        return [dict]

