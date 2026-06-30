from typing import List

from train.curriculum import OpponentDifficulty, OpponentStage
from train.curriculum import Curriculum


class PricingCurriculum(Curriculum):
    """Manages opponent difficulty progression."""
    def __init__(self) -> None:
        super().__init__()
        self.opponent_sequence = [
                            OpponentStage(
                                name = "premium_uniform",
                                opponent_type =  "premium_uniform",
                                description =  "Fixed price 3.5, easiest",
                                difficulty =  OpponentDifficulty.TUTORIAL,
                            ),
                            
                            OpponentStage(
                                name = "premium_bbp",
                                opponent_type =  "premium_bbp",
                                description =  "Fixed price_new= 3.0, price_old= 4.0, tutorial",
                                difficulty =  OpponentDifficulty.TUTORIAL,
                            ),

                            OpponentStage(
                                name = "premium_passive_uniform",
                                opponent_type =  "premium_passive_uniform",
                                description =  "Fixed price 3, between passive and premium",
                                difficulty =  OpponentDifficulty.TUTORIAL,
                            ),
                            
                            OpponentStage(
                                name = "premium_passive_bbp",
                                opponent_type =  "premium_passive_bbp",
                                description =  "Fixed price_new= 2.5, price_old= 3.5, tutorial",
                                difficulty =  OpponentDifficulty.TUTORIAL,
                            ),

                            OpponentStage(
                                name = "passive_uniform",
                                opponent_type =  "passive_uniform",
                                description =  "Fixed price 2.5, more competitive",
                                difficulty =  OpponentDifficulty.TUTORIAL,
                            ),
                            


                            

                            OpponentStage(
                                name = "passive_bbp",
                                opponent_type =  "passive_bbp",
                                description =  "Fixed price_new= 2.0, price_old= 3.0, tutorial",
                                difficulty =  OpponentDifficulty.TUTORIAL,
                            ),
                            
                            
                            OpponentStage(
                                name = "passive_aggressive_uniform",
                                opponent_type =  "passive_aggressive_uniform",
                                description =  "Fixed price 2.0, more competitive",
                                difficulty =  OpponentDifficulty.TUTORIAL,
                            ),
                            
                            
                            OpponentStage(
                                name = "passive_aggressive_bbp",
                                opponent_type =  "passive_aggressive_bbp",
                                description =  "Fixed price_new= 1.5, price_old= 2.5, tutorial",
                                difficulty =  OpponentDifficulty.TUTORIAL,
                            ),
                            
                            OpponentStage(
                                name = "aggressive_uniform",
                                opponent_type =  "aggressive_uniform",
                                description =  "Fixed price 1.5, most competitive",
                                difficulty =  OpponentDifficulty.TUTORIAL,
                            ),
                            
                            
                            OpponentStage(
                                name = "aggressive_bbp",
                                opponent_type =  "aggressive_bbp",
                                description =  "Fixed price_new= 1.0, price_old= 2.0, tutorial",
                                difficulty =  OpponentDifficulty.TUTORIAL,
                            ),

                            
                                ]
    def get_sequence(self) -> List[OpponentStage]:
        sequence = []
        for stage in self.opponent_sequence:
            if stage.opponent_type != "mixed":
                sequence.append(stage)
        return sequence
  
    @property
    def opponent_types(self) -> List[str]:
        opp_types = []
        for stage in self.opponent_sequence:
            if not stage.opponent_type == "mixed":
                opp_types.append(stage.opponent_type)   
        return opp_types



    