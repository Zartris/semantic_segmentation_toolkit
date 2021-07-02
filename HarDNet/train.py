import argparse
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp

import yaml
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.loader import get_loader
from ptsemseg.loss import get_loss_function, get_loss_class
from ptsemseg.metrics.metrics import runningScore, averageMeter
from models import get_model
from ptsemseg.optimizers import get_optimizer
from ptsemseg.schedulers import get_scheduler
from ptsemseg.utils import get_logger, convert_images
from torchsummary import summary


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)


def train(cfg, logger, logdir):
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))
    writer = SummaryWriter(log_dir=logdir)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["train_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=data_aug,
    )

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["val_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
    )

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
    )

    valloader = data.DataLoader(
        v_loader, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["n_workers"]
    )

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model = get_model(cfg["model"], n_classes).to(device)
    summary(model, (cfg["data"]["channels"], cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
            batch_size=cfg["training"]["batch_size"])
    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters:', total_params)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.apply(weights_init)
    pretrained_path = 'weights/hardnet_petite_base.pth'
    weights = torch.load(pretrained_path)
    model.module.base.load_state_dict(weights)

    # Setup optimizer, lr_scheduler and loss function
    scaler = amp.GradScaler()
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    print("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"], max_iterations=cfg["training"]["train_iters"])

    loss_fn = get_loss_class(cfg)
    print("Using loss {}".format(loss_fn))

    start_iter = 0
    best_iou = -100.0
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            best_iou = checkpoint["best_iou"] if 'best_iou' in checkpoint else -100.0
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    if cfg["training"]["finetune"] is not None:
        if os.path.isfile(cfg["training"]["finetune"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["finetune"])
            )
            checkpoint = torch.load(cfg["training"]["finetune"])
            model.load_state_dict(checkpoint["model_state"])

    val_loss_meter = averageMeter("val_loss")
    loss_meter = averageMeter('train_loss')
    time_meter = averageMeter("timer")

    index_to_category_name = [k for k in data_loader.categories.keys()]

    i = start_iter
    if hasattr(loss_fn, 'epoch'):
        loss_fn.epoch = start_iter
    flag = True
    loss_all = 0
    loss_n = 0
    while i <= cfg["training"]["train_iters"] and flag:
        for i_train_batch, (images, labels, _) in enumerate(trainloader):
            i += 1
            start_ts = time.time()
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with amp.autocast(enabled=cfg['training']['use_fp16']):
                outputs = model(images)
                loss = loss_fn(input=outputs, target=labels)

            if cfg['training']['use_fp16']:
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss).backward()
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            c_lr = scheduler.get_lr()
            scheduler.step()  # After optimizer.step

            loss_meter.update(loss.item())
            time_meter.update(time.time() - start_ts)
            loss_all += loss.item()
            loss_n += 1

            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}  lr={:.6f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss_all / loss_n,
                    time_meter.avg / cfg["training"]["batch_size"],
                    c_lr[0],
                )

                print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()

            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"]["train_iters"]:
                torch.cuda.empty_cache()
                model.eval()
                loss_all = 0
                loss_n = 0
                with torch.no_grad():
                    combined_img_list = []
                    for i_val, (images_val, labels_val, _) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)
                        with amp.autocast(enabled=cfg['training']['use_fp16']):
                            logits = model(images_val)
                            val_loss = loss_fn(input=logits, target=labels_val)

                        probs = torch.softmax(logits, dim=1)
                        pred = torch.argmax(probs, dim=1).cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())
                        combined_img_list = convert_images(writer, valloader, images_val.data.cpu().numpy(), pred, gt,
                                                           i_val, i + 1, combined_img_list, upload=False)

                # Add img
                # stack = np.stack(combined_img_list, axis=0)
                # writer.add_images("combined", stack, i + 1, dataformats="NHWC")

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
                logger.info("Iter %d Val Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                for k, v in class_iou.items():
                    logger.info("class_iou class {}: {}".format(index_to_category_name[k], v))
                    writer.add_scalar("val_metrics/cls_{}".format(index_to_category_name[k]), v, i + 1)

                logger.info("Best IoU: {}".format(best_iou))
                running_metrics_val.plot_conf_matrix(i + 1, index_to_category_name, writer,
                                                     plot_tile_suffix=str(score["Mean IoU : \t"]))

                better_iou = score["Mean IoU : \t"] >= best_iou
                best_iou = score["Mean IoU : \t"] if better_iou else best_iou
                state = {
                    "epoch": i + 1,
                    "best_iou": best_iou,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                }
                save_path = os.path.join(
                    writer.file_writer.get_logdir(),
                    "{}_{}_checkpoint.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                )
                torch.save(state, save_path)

                if better_iou:
                    logger.info("Saving model as the best")
                    print("Saving model as the best")
                    best_iou = score["Mean IoU : \t"]
                    running_metrics_val.plot_conf_matrix(i + 1, index_to_category_name, writer,
                                                         title_suffix=" (best)",
                                                         plot_tile_suffix=" " + str(best_iou),
                                                         save_image=writer.file_writer.get_logdir())
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "best_iou": best_iou,
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)
                    for image_index, img in enumerate(combined_img_list):
                        writer.add_image("(best) image_index: {}".format(str(image_index)),
                                         img, i + 1, dataformats='HWC')

                # Reset for next iteration:
                val_loss_meter.reset()
                running_metrics_val.reset()

                torch.cuda.empty_cache()

            if (i + 1) % cfg["training"]["log_splitter_inverval"] == 0:
                writer.flush()
                writer = SummaryWriter(log_dir=logdir)

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/hardnet.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    run_id = random.randint(1, 100000)
    # To prevent wrong paths if script is called from other folders.
    chwdir = os.path.dirname(os.path.realpath(__file__))
    logdir = os.path.join(chwdir, "runs", os.path.basename(args.config)[:-4], "cur")
    logdir_path = Path(str(logdir) + "/")
    if not logdir_path.exists():
        logdir_path.mkdir(parents=True)
    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    train(cfg, logger, logdir)
