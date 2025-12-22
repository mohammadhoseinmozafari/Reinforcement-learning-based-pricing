import random
import numpy as np
class Firm :
    def __init__(self, id , location, discount_factor, price_bounds, popularity = 0 ):
        """
        initialize a firm with given attributes.
        
        :param id: unique identifier for the firm.
        :param location: location of the firm on a linear market (ranging from 0 to 1).
        :param discount_factor: firm's discount factor (ranging from 0 to 1).
        
        """
        self.id = id
        assert 0.0 <= location <= 1.0, "Location must be between 0 and 1"
        self.location = location 
        assert 0.0 <= discount_factor <= 1.0, "Discount factor must be between 0 and 1"
        self.discount_factor= discount_factor
        self.price_bounds = price_bounds  # (min_price, max_price)
        self.popularity = popularity
        self.price_history = [[0,0,0]]  # (P_uniform, P_old, P_new)
        self.profit_history = [0.0]
        self.popularity_history = [0]

    def get_firm_info (self) :
        print("Firm ID: ", self.id)
        print("Firm Location: ", self.location)
        print("Firm Discount Factor: ", self.discount_factor)
        print("Firm Price Bounds: ", self.price_bounds)
        print("Firm Popularity: ", self.popularity)
        print("Firm Price History: ", self.price_history)
        print("Firm Profit History: ", self.profit_history)
        print("Firm Popularity History: ", self.popularity_history)


    def get_price(self):
        return self.price_history[-1] if self.price_history else None
    
    def get_popularity(self):
        return self.popularity
    
    def update_popularity(self, change):
        self.popularity += change
        self.popularity_history.append(self.popularity)

    def get_price_trend(self):
        return self.price_history if self.price_history else None
    
    def get_popularity_trend(self):
        return self.popularity_history if self.popularity_history else None
    
    def choose_price(self, observation):
        """
        Placeholder method for choosing price based on observation.
        :param observation: current market observation.
        :return: chosen price within price bounds.

        """
        min_price, max_price = self.price_bounds
        chosen_price = random.uniform(min_price, max_price)
        self.price_history.append(chosen_price)
        return chosen_price
    
