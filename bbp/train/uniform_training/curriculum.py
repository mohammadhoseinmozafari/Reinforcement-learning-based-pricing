from train.curriculum import OpponentDifficulty, OpponentStage

class UniformPricingCurriculum:
    """Manages opponent difficulty progression."""
    
    OPPONENT_SEQUENCE = [
        OpponentStage(
            name = "premium_uniform",
            opponent_type =  "premium_uniform",
            description =  "Fixed price 3.5, never changes",
            difficulty =  OpponentDifficulty.TUTORIAL,
        ),
        OpponentStage(
            name = "passive_uniform",
            opponent_type =  "passive_uniform",
            description =  "Fixed price 2.5, more competitive",
            difficulty =  OpponentDifficulty.TUTORIAL,
        ),
        OpponentStage(
            name = "aggressive_uniform",
            opponent_type =  "aggressive_uniform",
            description =  "Fixed price 1.5, most competitive",
            difficulty =  OpponentDifficulty.EASY,
        ),

        OpponentStage(
            name = "premium_random_reactive_uniform",
            opponent_type= "premium_random_reactive_uniform",
            description =  "Creates random premium opponets which are reactive",
            difficulty =  OpponentDifficulty.MEDIUM,
        ),
     
        OpponentStage(
            name = "passive_random_reactive_uniform",
            opponent_type= "passive_random_reactive_uniform",
            description =  "Creates random passive opponets which are reactive",
            difficulty =  OpponentDifficulty.HARD,
        ),
        OpponentStage(
            name = "aggressive_random_reactive_uniform",
            opponent_type= "aggressive_random_reactive_uniform",
            description =  "Creates random aggressive opponets which are reactive",
            difficulty =  OpponentDifficulty.HARD,
        ),
        
        
   ]
    
  
    
