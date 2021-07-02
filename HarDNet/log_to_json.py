import json
from pathlib import Path


def convert_tensorboard_log_to_json(log_dir_path: Path):
    result = {}
    training_info = {}
    val_info = {}
    log_files = [str(log) for log in log_dir_path.glob("*.log")]
    log_files = sorted(log_files)
    step_size = 10
    step = 0
    skip_lines = 0
    for logfile in log_files:
        with open(logfile) as f:
            f = f.readlines()

        for i, line in enumerate(f):
            debug = 0
            if skip_lines > 0:
                skip_lines -= 1
                continue
            if "INFO Iter [" in line:
                line_info = {}
                start_index = line.find("[")
                mid_index = line.find("]")
                step = int(line[start_index + 1:mid_index].split("/")[0])
                info = line[mid_index + 3:-1].split('  ')
                line_info["Loss"] = float(info[0].split(": ")[-1])
                line_info["lr"] = float(info[-1].split("=")[-1])

                training_info[step] = line_info
                # Training step:
            elif "Val Loss" in line:
                line_info = {}
                start_index = line.find("Loss:")
                skip_lines = 8
                line_info["Loss"] = float(line[start_index: -1].split(": ")[-1])
                next_line = f[i + 1]
                line_info["Overall Acc"] = float(next_line.split(":")[-1].replace(" ", ""))
                next_line = f[i + 2]
                line_info["Mean Acc"] = float(next_line.split(":")[-1].replace(" ", ""))
                next_line = f[i + 3]
                line_info["FreqW Acc"] = float(next_line.split(":")[-1].replace(" ", ""))
                next_line = f[i + 4]
                line_info["Mean IoU"] = float(next_line.split(":")[-1].replace(" ", ""))
                class_iou = {}
                next_line = f[i + 5]
                class_iou["unknown"] = float(next_line.split(":")[-1].replace(" ", ""))
                next_line = f[i + 6]
                class_iou["water"] = float(next_line.split(":")[-1].replace(" ", ""))
                next_line = f[i + 7]
                class_iou["ship"] = float(next_line.split(":")[-1].replace(" ", ""))
                next_line = f[i + 8]
                line_info["Best IoU"] = float(next_line.split(":")[-1].replace(" ", ""))
                line_info["Class IoU"] = class_iou
                val_info[step] = line_info
    result["train_info"] = training_info
    result["val_info"] = val_info
    with open(str(Path(log_dir_path, "train_info.json")), 'w') as fp:
        json.dump(result, fp)
    debug = 0


if __name__ == '__main__':
    log_folder = Path("/FCHarDNet/runs/hardnet/cur")
    convert_tensorboard_log_to_json(log_folder)
