from pathlib import Path

import torch
import argparse
import os
import numpy as np
import yaml
from PIL import Image

from models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict

torch.backends.cudnn.benchmark = True
import cv2


def init_model(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = get_loader(cfg["data"]["dataset"])
    loader = data_loader(
        root=None,
        is_transform=True,
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        test_mode=True
    )
    n_classes = loader.n_classes

    # Setup Model
    model = get_model({"arch": "bisenetv2"}, n_classes)
    model.validate_mode()
    def weights_init(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
    model.apply(weights_init)
    checkpoint = torch.load(cfg["validate"]["model_path"])
    state = convert_state_dict(checkpoint["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    return device, model, loader


def run_model(args, cfg):
    size = (cfg['data']['img_rows'], cfg['data']['img_cols'])
    device, model, loader = init_model(args, cfg)
    proc_size = size
    handle_file(Path(args.input), Path(args.output), proc_size, device, model, loader)

    # if os.path.isfile(args.input):
    #     img_raw, decoded = process_img(args.input, proc_size, device, model, loader)
    #     blend = np.concatenate((img_raw, decoded), axis=1)
    #     out_path = os.path.join(args.output, os.path.basename(args.input))
    #     cv2.imwrite("test.png", decoded)
    #     cv2.imwrite(out_path, blend)
    #
    # elif os.path.isdir(args.input):
    #     print("Process all image inside : {}".format(args.input))
    #
    #     for img_file in os.listdir(args.input):
    #         _, ext = os.path.splitext(os.path.basename((img_file)))
    #         if ext not in [".png", ".jpg"]:
    #             continue
    #         img_path = os.path.join(args.input, img_file)
    #
    #         img, decoded = process_img(img_path, proc_size, device, model, loader)
    #         blend = np.concatenate((img, decoded), axis=1)
    #         out_path = os.path.join(args.output, os.path.basename(img_file))
    #         cv2.imwrite(out_path, blend)


def handle_file(file_path: Path, output_path: Path, proc_size, device, model, loader):
    if file_path.is_dir():
        print("Process all image inside : {}".format(file_path.stem))
        sub_output = Path(output_path, file_path.stem)
        if not sub_output.exists():
            sub_output.mkdir(parents=True)
        for sub_file in file_path.iterdir():
            handle_file(sub_file, sub_output, proc_size, device, model, loader)

    if file_path.suffix in [".png", ".jpg"]:
        # print("Process image: {}".format(file_path.stem))
        img_raw, decoded = process_img(file_path, proc_size, device, model, loader)
        blend = np.concatenate((img_raw, decoded), axis=1)
        cv2.imwrite(str(Path(output_path, file_path.stem + "_pred.png")), decoded)
        cv2.imwrite(str(Path(output_path, file_path.stem + "_blend.png")), blend)


def process_img(img_path, size, device, model, loader):
    print("Read Input Image from : {}".format(str(img_path)))
    img = Image.open(str(img_path))
    img = np.array(img, dtype=np.uint8)
    if img.shape[-1] == 4:  # check if RGBA
        img = img[:, :, :3]  # Remove alpha
    img = img.astype(np.uint8)
    img, _ = loader.transform(img, None)
    img = img.unsqueeze(0)  # Insert batch size of 1

    images = img.to(device)
    outputs = model(images)

    pred = np.squeeze(outputs.cpu().numpy(), 0)
    decoded = loader.decode_segmap(pred)
    img = img.cpu().numpy().squeeze(0)
    img = img.transpose(1, 2, 0) # HWC
    return img * 255, decoded * 255


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--config_path",
        type=str,
        default="F:\\code\\python\\Hardnet\\BiSeNet\\runs\\BiSeNetv2_dry\\cur\\BiSeNetv2_dry.yml",
        help="The config path is used to extract specific model parameters",
    )
    parser.add_argument(
        "--input", nargs="?", type=str, default="D:\\Datasets\\DryHarbour\\images_fayard",
        help="Path of the input image/ directory"
    )
    parser.add_argument(
        "--output", nargs="?", type=str, default="F:\\code\\python\\Hardnet\\BiSeNet\\runs\\BiSeNetv2_dry\\cur\\test",
        help="Path of the output directory"
    )
    args = parser.parse_args()

    with open(args.config_path) as fp:
        cfg = yaml.load(fp)

    run_model(args, cfg)
