import copy

from BiSeNet.models.bisenetv1 import BiSeNetV1
from BiSeNet.models.bisenetv2 import BiSeNetV2


def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    model = model(n_classes=n_classes, **param_dict)
    return model


def _get_model_instance(name):
    try:
        return {
            "bisenetv1": BiSeNetV1,
            "bisenetv2": BiSeNetV2
        }[name]
    except:
        raise ("Model {} not available".format(name))
