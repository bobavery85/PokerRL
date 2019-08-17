import ray
import torch

from PokerRL.eval.head_to_head.LocalHistoryMaster import LocalHistoryMaster as LocalEvalHistoryMaster


@ray.remote(num_cpus=1, num_gpus=1 if torch.cuda.is_available() else 0)
class DistHistoryMaster(LocalEvalHistoryMaster):

    def __init__(self, t_prof, chief_handle, eval_agent_cls):
        LocalEvalHistoryMaster.__init__(self, t_prof=t_prof, chief_handle=chief_handle, eval_agent_cls=eval_agent_cls)
