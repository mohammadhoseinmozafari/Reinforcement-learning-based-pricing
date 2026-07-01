
from enum import Enum

class Stage(str, Enum):
    TRAIN   =   "train"
    EVAL    =   "eval"

class Phase (str, Enum):
    UNIFORM_PRICING =   "uniform_pricing"
    BBP_PRICING     =   "bbp_pricing"

class OpponentType(str, Enum):
    BBP = 'bbp_opp'
    UNIFORM = 'uniform_opp'

class PathResolver :
    def __init__(self) -> None:
        pass            
         

    def resolve_train_path(self, phase: str, opponent_type : str, run: int ) -> str:
        phase = self._resolve_phase(phase)
        opponent_type = self._resolve_opp_type(opponent_type)
        return f"experiments/{phase}_vs_{opponent_type}/runs/{run}"
    
    def resolve_eval_path (self , phase: str , opponent_type : str) -> str:
        phase = self._resolve_phase(phase)
        opponent_type = self._resolve_opp_type(opponent_type)
        return f"experiments/{phase}_vs_{opponent_type}/eval"


    def _resolve_stage (self, stage : str) -> Stage:
        if stage == "train":
            return Stage.TRAIN
        elif stage == "eval":
            return Stage.EVAL
        
        else:
            raise ValueError
        
    def _resolve_phase (self, phase: str) -> str:
        if phase =="Uniform Pricing":
            return "uniform"
        elif phase == "Behavior Based Pricing":
            return "bbp"
        
        else:
            raise ValueError
        
    def _resolve_opp_type (self, opp_type: str)->str:

        if opp_type == "Uniform Pricing Opponent":
            return "uniform"
        
        elif opp_type == "BBP Opponent":
            return "bbp"
        
        else :
            raise ValueError

    
