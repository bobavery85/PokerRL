import ray
import torch

from PokerRL.eval.head_to_head.LocalOfflineMaster import LocalOfflineMaster as LocalEvalOfflineMaster


@ray.remote(num_cpus=1, num_gpus=1 if torch.cuda.is_available() else 0)
class DistOfflineMaster(LocalEvalOfflineMaster):

    def __init__(self, t_prof, chief_handle, eval_agent_cls):
        LocalEvalOfflineMaster.__init__(self, t_prof=t_prof, chief_handle=chief_handle, eval_agent_cls=eval_agent_cls)
