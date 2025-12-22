import pdb
import functools
import pandas as pd
from firm import Firm
from consumer import Consumer
import numpy as np
from copy import copy
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv

P_U = 2
P_O = 2.6
P_N = 1.4
class BBPHottelingEnv(ParallelEnv) :
    
    metadata = {
        "name": "BBPHottelingEnv_v0",
    }
    def __init__(self, n_consumer=1,horizon = 100, price_bounds= (0, 5),t_cost = 1,v= 5):
        self.possible_agents = ["firm_A","firm_B"]
        self.agent_name_mapping = {"firm_A": "A", "firm_B": "B"}

        self.n_consumers = n_consumer
        self.horizon = horizon
        self.t_cost = t_cost
        self.v = v
        self.p_min ,self.p_max = price_bounds
        self._build_firms()
        self._build_consumers()

        self.firm_df = pd.DataFrame(columns=['time_step', 'firm_id', 'location', 'discount_factor', 'price_bounds', 'popularity', 'price'])
        self.consumer_df = pd.DataFrame(columns=['time_step', 'consumer_id', 'location', 'exclusivity_sensitivity', 'strategicness', 'T', 'V','last_choice'])
        


        
        


    def _build_firms(self):
        self.firms = {
            "A" : Firm(id="A",location=0,discount_factor=0.95, price_bounds=(self.p_min, self.p_max)),
            "B" : Firm(id="B",location=1,discount_factor=0.95, price_bounds=(self.p_min, self.p_max)),
        }

    def _build_consumers(self):
        self.consumers= []

        for n in range(self.n_consumers):
            location = np.random.uniform(0,1)
            exclusivity_sensitivity = np.random.uniform(0,1)
            strategicness = np.random.uniform(0,1)
            self.consumers.append(

                Consumer(location=location,
                         exclusivity_sensitivity= exclusivity_sensitivity,
                        strategicness=strategicness,T=self.t_cost, V=self.v)
                        
                                )
    def reset(self, seed = None, options = None):
        self.debug_log = {}
        self.agents = copy (self.possible_agents)
        self.time_step = 0
        # reset firms
        for firm in self.firms.values():
            firm.popularity = 0
            firm.price_history = [[0,0,0]],
            firm.popularity_history = [0],
            firm.profit_history = [0.0],
        # reset consumers
        for c in self.consumers:
            c.purchase_history = [0],
            c.last_choice = None

        
        observations = {agent: self._get_obs() for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        # logging initial state
        for firm_id, firm in self.firms.items():
            self.firm_df = pd.concat([self.firm_df, pd.DataFrame([{
                'time_step': self.time_step,
                'firm_id': firm.id,
                'location': firm.location,
                'discount_factor': firm.discount_factor,
                'price_bounds': firm.price_bounds,
                'popularity': firm.popularity,
                'price': firm.get_price()
            }])], ignore_index=True)
        for consumer_id, consumer in enumerate(self.consumers):
            self.consumer_df = pd.concat([self.consumer_df, pd.DataFrame([{
                'time_step': self.time_step,
                'consumer_id': consumer_id,
                'location': consumer.location,
                'exclusivity_sensitivity': consumer.exclusivity_sensitivity,
                'strategicness': consumer.strategicness,
                'T': consumer.T,
                'V': consumer.V,
                'last_choice': consumer.last_choice
            }])], ignore_index=True)

        print ("Environment reset complete.")
        print ("Initial Firm States:\n", self.firm_df)
        print ("Initial Consumer States:\n", self.consumer_df)

        return observations, infos
        

    def step(self, actions):
        self.time_step +=1 

        for agent, action in actions.items():
            firm_id = self.agent_name_mapping[agent]
            firm = self.firms[firm_id]

            mode, p_u, p_o, p_n = action

            if self.time_step == 1:
                p_o, p_n = p_u, p_u
            firm.price_history.append([p_u, p_o, p_n])
        # market clearing
        demand = {
            "A" :[],
            "B" : []
        }
        for firm in self.firms.values():
            firm.popularity = 0
        
        for consumer in self.consumers:
            faced_prices = {}

            for firm_id, firm in self.firms.items():
                p_u, p_o, p_n = firm.get_price()

                if consumer.last_choice == firm_id:
                    faced_prices[firm_id] = p_o
                else :
                    faced_prices[firm_id] = p_n

                firm._current_price_for_consumer = faced_prices[firm_id]

            def _temp_get_price(firm=firm):
                return firm._current_price_for_consumer
            
            for firm in self.firms.values():
                firm.get_price = _temp_get_price
            
            choice = consumer.choose_firm([self.firms["A"],self.firms["B"]])
            demand[choice].append(faced_prices[choice])

        rewards = {}
        for firm_id , agent in zip(["A","B"],self.agents):
            profit = sum(demand[firm_id]/self.n_consumers)
            self.firms[firm_id].profit_history[self.time_step] = profit
            rewards[agent] = profit
        for firm in self.firms.values():
            firm.popularity_history[self.time_step] = firm.popularity
        
        truncations = {agent: self.time_step >= self.horizon for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        observations = {agent: self._get_obs() for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if all(truncations.values()):
            self.agents = []
        return observations, rewards, terminations, truncations, infos
    
    def _get_obs(self):
        A, B = self.firms["A"], self.firms["B"]
        
        
        return None

        # return np.array(
        #     [
        #         A.get_price(),
        #         B.get_price(),
        #         A.popularity,
        #         B.popularity,
        #     ]
            
        # )
    
    
    def render(self):
        print (
            f"Time step: {self.time_step}\n"
            f"Profit A: {self.firms['A'].profit_history[self.time_step]}\n"
            f"Profit B: {self.firms['B'].profit_history[self.time_step]}\n"
            f"Popularity A: {self.firms['A'].popularity_history[self.time_step]}\n"
            f"Popularity B: {self.firms['B'].popularity_history[self.time_step]}\n"
        )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=10, shape=(8,), dtype=np.float32)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(
            low = np.array([0, *self.firms["A"].price_bounds], *self.firms["A"].price_bounds),
            high = np.array([1, *self.firms["A"].price_bounds][::-1], *self.firms["A"].price_bounds[::-1]),
            dtype=np.float32
        )

    


env = BBPHottelingEnv()
observations, infos = env.reset()
