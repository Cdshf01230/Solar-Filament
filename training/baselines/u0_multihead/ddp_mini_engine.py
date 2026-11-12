"""Use the existing DDP trainer with the smaller head and explicit protocol."""

from . import ddp_engine as runner
from .ddp_mini import DDPModel, PROTOCOL

runner.SharedFrozenModel = DDPModel
runner.DDP_PROTOCOL = PROTOCOL
if __name__ == '__main__':
    runner.main()
