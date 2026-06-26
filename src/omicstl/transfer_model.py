"""Transfer model wrapper class"""

from typing import Literal

from omicstl.transfer_forest import TransferForest
from omicstl.transfer_networks import TransferMLP, TransferVAE

class TransferModel:
    type: Literal['dl', 'rf']
    model: TransferForest | TransferMLP | TransferVAE

    def __init__(self, model: TransferForest | TransferMLP | TransferVAE):
        self.model = model
        self.type = 'rf' if type(model) is TransferForest else 'dl'