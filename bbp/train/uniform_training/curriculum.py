from typing import List

from train.curriculum import OpponentDifficulty, OpponentStage
from train.curriculum import Curriculum


class UniformPricingCurriculum(Curriculum):
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
                                name = "premium_passive_uniform",
                                opponent_type =  "premium_passive_uniform",
                                description =  "Fixed price 3, between passive and premium",
                                difficulty =  OpponentDifficulty.TUTORIAL,
                            ),

                            OpponentStage(
                                name = "passive_uniform",
                                opponent_type =  "passive_uniform",
                                description =  "Fixed price 2.5, more competitive",
                                difficulty =  OpponentDifficulty.TUTORIAL,
                            ),
                            
                            OpponentStage(
                                name = "passive_aggressive_uniform",
                                opponent_type =  "passive_aggressive_uniform",
                                description =  "Fixed price 2.0, more competitive",
                                difficulty =  OpponentDifficulty.TUTORIAL,
                            ),
                            
                            
                            OpponentStage(
                                name = "aggressive_uniform",
                                opponent_type =  "aggressive_uniform",
                                description =  "Fixed price 1.5, most competitive",
                                difficulty =  OpponentDifficulty.TUTORIAL,
                            ),

                            # OpponentStage(
                            #     name = "premium_uniform",
                            #     opponent_type =  "premium_uniform",
                            #     description =  "Fixed price 4, never changes",
                            #     difficulty =  OpponentDifficulty.TUTORIAL,
                            # ),


                            # OpponentStage(
                            #     name="constant_opponent_mixed",
                            #     opponent_type="mixed",
                            #     opponent_types = ['premium_uniform', "passive_uniform", "aggressive_uniform"],
                            #     description="Mixes all constant opponent policies",
                            #     difficulty=OpponentDifficulty.EASY

                            # )

                            # OpponentStage(
                            #     name = "premium_random_reactive_uniform",
                            #     opponent_type= "premium_random_reactive_uniform",
                            #     description =  "Creates random premium opponets which are reactive",
                            #     difficulty =  OpponentDifficulty.MEDIUM,
                            # ),
                        
                            # OpponentStage(
                            #     name = "passive_random_reactive_uniform",
                            #     opponent_type= "passive_random_reactive_uniform",
                            #     description =  "Creates random passive opponets which are reactive",
                            #     difficulty =  OpponentDifficulty.HARD,
                            # ),
                            # OpponentStage(
                            #     name = "aggressive_random_reactive_uniform",
                            #     opponent_type= "aggressive_random_reactive_uniform",
                            #     description =  "Creates random aggressive opponets which are reactive",
                            #     difficulty =  OpponentDifficulty.HARD,
                            # ),
                            
                                ]
    def get_sequence(self) -> List[OpponentStage]:
        sequence = []
        for stage in self.opponent_sequence:
            if stage.opponent_type != "mixed":
                sequence.append(stage)
        return sequence
  

    
  
    
