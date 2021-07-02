import copy

from DDRNet.models.DDRNet_39 import DualResNet as DDRNet_39
from DDRNet.models.DDRNet_23 import DualResNet as DDRNet_23
from DDRNet.models.DDRNet_23_slim import DualResNet as DDRNet_23_slim


def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")
    # Check for block type:
    model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "DDRNet_39": DDRNet_39,
            "DDRNet_23": DDRNet_23,
            "DDRNet_23_slim": DDRNet_23_slim,
        }[name]
    except:
        raise ("Model {} not available".format(name))
