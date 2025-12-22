

class Consumer :
    def __init__(self, location , exclusivity_sensitivity , strategicness, T, V):
        """
        initialize a consumer with given attributes.
        
        :param id: id of the consumer
        :param location: location of the consumer (ranging from 0 to 1)
        :param exclusivity_sensitivity: sensitivity to exclusivity (ranging from 0 to 1).
        :param strategicness: strategicness level of the consumer (ranging from 0 to 1).

        """
        assert 0.0 <= location <= 1.0, "Location must be between 0 and 1"
        self.location = location
        assert 0.0 <= exclusivity_sensitivity <= 1.0, "Exclusivity sensitivity must be between 0 and 1"
        self.exclusivity_sensitivity = exclusivity_sensitivity
        assert 0.0 <= strategicness <= 1.0, "Strategicness must be between 0 and 1"
        self.strategicness = strategicness
        assert T >= 0.0, "Time cost parameter T must be non-negative"
        self.T = T
        assert V >= 0.0, "Value V must be non-negative"
        self.V = V
        self.strategicness_type = self.get_strategicness_type()
        self.purchase_history = []
        self.last_choice = None

    def get_consumer_info (self):
        print ("Consumer location: ", self.location)
        print("Consumer Exclusivity sensitivity: " , self.exclusivity_sensitivity)
        print("Consumer Strategicness: ",self.strategicness )
        print("Consumer V: ",self.V)
        print("Consumer T : ",self.T)
        print("Consumer Purchase History : ", self.purchase_history)
        print("consumer last Choice: ",self.last_choice)

    def get_strategicness_type(self):
        if self.strategicness < 0.3:
            return 'myopic'
        elif self.strategicness < 0.7:
            return 'balanced'
        else:
            return 'strategic'
    

    def get_mismatch_cost(self, firm):
        if firm.location is not None:
            return abs(self.location - firm.location)
        else:
            return None
    def get_instant_utility(self, firm):
        mismatch_cost = self.get_mismatch_cost(firm)
        if mismatch_cost is None:
            return None
        instant_utility = self.V - firm.get_price() - (self.T * mismatch_cost) - (self.exclusivity_sensitivity * firm.get_popularity())
        return instant_utility
    
    def get_expected_future_price(self, firm) : 
        """
        The consumer approximates next period price based on the firm's price trend with a moving average.
        :param firm: the firm whose price trend is being considered.
        :return: expected future price or None if price trend is not available.
        """
        price_trend = firm.get_price_trend() # Returns the price trend as a list of past prices.
        
        window_size = min(3, len(price_trend))
        expected_future_price = sum(price_trend[-window_size:]) / window_size
        return expected_future_price
    
    def get_expected_future_popularity(self, firm) : 
        """
        The consumer approximates next period popularity based on the firm's popularity trend with a moving average.
        :param firm: the firm whose popularity trend is being considered.
        :param current_popularity: the current popularity of the firm.
        :return: expected future popularity or None if popularity trend is not available.
        """
        popularity_trend = firm.get_popularity_trend() # Returns the popularity trend as a list of past popularities.
        
        
        window_size = min(3, len(popularity_trend))
        expected_future_popularity = sum(popularity_trend[-window_size:]) / window_size
        return expected_future_popularity

    def get_expected_future_utility(self, firm):
        """
        The consumer approximates next period utility based on expected future price and popularity.    
        :param firm: the firm whose expected future utility is being considered.
        :param price: the current price of the firm.
        :param popularity: the current popularity of the firm.
        :param V: the value of the product to the consumer.
        :param T: the time cost parameter.
        :return: expected future utility or None if mismatch cost cannot be computed.
        """
        expected_future_price = self.get_expected_future_price(firm)
        expected_future_popularity = self.get_expected_future_popularity(firm)
        mismatch_cost = self.get_mismatch_cost(firm)
        if mismatch_cost is None:
            return None 

        expected_future_utility = self.V - expected_future_price - (self.T * mismatch_cost) + (self.exclusivity_sensitivity * expected_future_popularity) 
        return expected_future_utility
   
    
    def choose_firm (self,firms):
        firmA , firmB= firms 
        utility_A = self.get_instant_utility(firmA)
        utility_B = self.get_instant_utility(firmB)

        expected_utility_A = self.get_expected_future_utility(firmA)
        expected_utility_B = self.get_expected_future_utility(firmB)
        
        utility_A += self.strategicness * expected_utility_A
        utility_B += self.strategicness * expected_utility_B
        if utility_A > utility_B:
            choice = firmA.id
            firmA.update_popularity(1)
        else:
            choice = firmB.id
            firmB.update_popularity(1)

        
        self.last_choice = choice
        self.purchase_history.append(choice)
        return choice
    


