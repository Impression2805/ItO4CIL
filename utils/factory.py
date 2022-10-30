from models.ucir import UCIR
from models.ucir_lto import UCIR_lto
from models.icarl import iCaRL
from models.replay import Replay
from models.bic import BiC
from models.podnet import PODNet
from models.podnet_lto import PODNet_lto
from models.wa import WA


def get_model(model_name, args):
    name = model_name.lower()
    if name == 'icarl':
        return iCaRL(args)
    elif name == 'bic':
        return BiC(args)
    elif name == 'podnet':
        return PODNet(args)
    elif name == 'podnet_lto':
        return PODNet_lto(args)
    elif name == "wa":
        return WA(args)
    elif name == "replay":
        return Replay(args)
    elif name == "ucir":
        return UCIR(args)
    elif name == "ucir_lto":
        return UCIR_lto(args)
    else:
        assert 0
