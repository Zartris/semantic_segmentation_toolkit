from pathlib import Path

import torch
import yaml
from torch import nn

from BiSeNet.models import get_model
from ptsemseg.utils import convert_state_dict


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)


def load_model(cfg, n_classes, out_path):
    model = get_model(cfg["model"], n_classes)
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.apply(weights_init)
    checkpoint = torch.load(str(cfg["model_path"]))
    state = convert_state_dict(checkpoint["model_state"])
    model.load_state_dict(state)
    m = torch.jit.script(model)
    torch.jit.save(m, str(out_path))


if __name__ == '__main__':
    file_path = Path("F:\\code\\python\\Hardnet\\BiSeNet\\runs\\BiSeNetv2_dry\\cur")
    file_path_cfg = Path(file_path, "BiSeNetv2_dry.yml")
    file_path_model = Path(file_path, "bisenetv2_dryharbour_5_best_model.pkl")
    out_path = Path(file_path, "bisenetv2_dryharbour_5_best_model.pt")
    n_classes = 5
    with open(str(file_path_cfg)) as fp:
        cfg = yaml.load(fp)
    cfg["model_path"] = file_path_model
    model = load_model(cfg, n_classes, out_path)
