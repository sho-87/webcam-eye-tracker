import ast
import random
import json
import datetime
import itertools
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from pathlib import Path
from collections import OrderedDict
from configparser import ConfigParser
from tqdm.autonotebook import tqdm
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
from ray import tune
from ray.tune import JupyterNotebookReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Analysis
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning.loggers import TensorBoardLogger
from Models import create_datasets, FaceDataset, SingleModel, EyesModel, FullModel


def get_config(path="config.ini", comment_char=";"):
    config_file = ConfigParser(inline_comment_prefixes=comment_char)
    config_file.read(path)

    config_default = config_file["DEFAULT"]
    config_colours = config_file["COLOURS"]
    config_eyetracker = config_file["EYETRACKER"]
    config_tf = config_file["TF"]

    settings = {key: ast.literal_eval(config_default[key]) for key in config_default}
    colours = {key: ast.literal_eval(config_colours[key]) for key in config_colours}
    eyetracker = {
        key: ast.literal_eval(config_eyetracker[key]) for key in config_eyetracker
    }
    tf = {key: ast.literal_eval(config_tf[key]) for key in config_tf}

    return settings, colours, eyetracker, tf


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((5, 2), dtype=dtype)
    for i in range(0, 5):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def bgr_to_rgb(img):
    """Convert from opencv BGR to RGB"""
    return img[..., ::-1].copy()


def clamp_value(x, max_value):
    """Restrict values to a range"""
    if x < 0:
        return 0
    if x > max_value:
        return max_value
    return x


def plot_region_map(path, region_map, map_scale, cmap="inferno"):
    """Create plot of number of data samples at each screen coordinate"""
    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(
        np.transpose(region_map).repeat(map_scale, axis=0).repeat(map_scale, axis=1),
        interpolation="bicubic",
        cmap=cmap,
    )
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    plt.colorbar(
        im, ticks=np.arange(np.min(region_map), np.max(region_map) + 1), cax=cax
    )
    ax.set_title("Number of samples at each screen region")
    plt.savefig(path)


def get_calibration_zones(w, h, target_radius):
    """Get coordinates for 9 point calibration"""
    xs = (0 + target_radius, w // 2, w - target_radius)
    ys = (0 + target_radius, h // 2, h - target_radius)
    zones = list(itertools.product(xs, ys))
    random.shuffle(zones)
    return zones


def get_undersampled_region(region_map, map_scale):
    """Get screen coordinates with fewest data samples"""
    min_coords = np.where(region_map == np.min(region_map))
    idx = random.randint(0, len(min_coords[0]) - 1)
    return (min_coords[0][idx] * map_scale, min_coords[1][idx] * map_scale)


def train_single(
    config,
    cwd,
    data_partial,
    img_types,
    num_epochs=1,
    num_gpus=-1,
    save_checkpoints=False,
):
    pl.seed_everything(config["seed"])

    d_train, d_val, d_test = create_datasets(
        cwd, data_partial, img_types, seed=config["seed"], batch_size=config["bs"]
    )

    model = SingleModel(config, *img_types)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        accelerator="dp",
        progress_bar_refresh_rate=0,
        checkpoint_callback=save_checkpoints,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version=".", log_graph=True
        ),
        callbacks=[TuneReportCallback({"loss": "val_loss"}, on="validation_end")],
    )

    trainer.fit(model, train_dataloader=d_train, val_dataloaders=d_val)


def train_eyes(
    config,
    cwd,
    data_partial,
    img_types,
    num_epochs=1,
    num_gpus=-1,
    save_checkpoints=False,
):
    pl.seed_everything(config["seed"])

    d_train, d_val, d_test = create_datasets(
        cwd, data_partial, img_types, seed=config["seed"], batch_size=config["bs"]
    )

    model = EyesModel(config)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        accelerator="dp",
        progress_bar_refresh_rate=0,
        checkpoint_callback=save_checkpoints,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version=".", log_graph=True
        ),
        callbacks=[TuneReportCallback({"loss": "val_loss"}, on="validation_end")],
    )

    trainer.fit(model, train_dataloader=d_train, val_dataloaders=d_val)


def train_full(
    config,
    cwd,
    data_partial,
    img_types,
    num_epochs=1,
    num_gpus=-1,
    save_checkpoints=False,
):
    pl.seed_everything(config["seed"])

    d_train, d_val, d_test = create_datasets(
        cwd, data_partial, img_types, seed=config["seed"], batch_size=config["bs"]
    )

    model = FullModel(config)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        accelerator="dp",
        progress_bar_refresh_rate=0,
        checkpoint_callback=save_checkpoints,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version=".", log_graph=True
        ),
        callbacks=[TuneReportCallback({"loss": "val_loss"}, on="validation_end")],
    )

    trainer.fit(model, train_dataloader=d_train, val_dataloaders=d_val)


def dir_name_string(trial):
    name = str(trial.experiment_tag)

    if len(name) > 100:
        return name[:100]
    else:
        return name


def tune_asha(
    config,
    train_func,
    name,
    img_types,
    num_samples,
    num_epochs,
    data_partial=False,
    save_checkpoints=False,
    seed=1,
):
    cwd = Path.cwd()
    random.seed(seed)
    np.random.seed(seed)

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    reporter = JupyterNotebookReporter(
        overwrite=True,
        parameter_columns=list(config.keys()),
        metric_columns=["loss", "training_iteration"],
    )

    analysis = tune.run(
        tune.with_parameters(
            train_func,
            cwd=cwd,
            data_partial=data_partial,
            img_types=img_types,
            save_checkpoints=save_checkpoints,
            num_epochs=num_epochs,
            num_gpus=1,
        ),
        resources_per_trial={"cpu": 2, "gpu": 1},
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        max_failures=1,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="{}/{}".format(
            name, datetime.datetime.now().strftime("%Y-%b-%d %H-%M-%S")
        ),
        trial_dirname_creator=dir_name_string,
        local_dir=cwd / "logs",
        raise_on_failed_trial=False,
        verbose=3,
    )

    print("Best hyperparameters: {}".format(analysis.best_config))

    return analysis


def get_tune_results(analysis):
    """Get results from single experiment"""

    if analysis.best_checkpoint:
        print(f"Directory: {analysis.best_checkpoint}")
    else:
        print(f"Directory: {analysis.best_logdir}")

    print(f"Loss: {round(analysis.best_result['loss'],2)}")
    print(f"Pixel error: {round(np.sqrt(analysis.best_result['loss']),2)}")
    print("Hyperparameters...")
    for hparam in analysis.best_config:
        print(f"- {hparam}: {analysis.best_config[hparam]}")


def get_best_results(path):
    """Get best results in a directory"""
    analysis = Analysis(path, default_metric="loss", default_mode="min")
    df = analysis.dataframe()
    df.sort_values("loss", inplace=True)
    best = df.head(1)

    print(f"\n--- Best of '{path}' ---\n")
    print(f"Directory: {best['logdir'].values[0]}")
    print(f"Loss: {round(best['loss'].values[0],2)}")
    print(f"Pixel error: {round(np.sqrt(best['loss'].values[0]),2)}")

    hyperparams = best.filter(like="config", axis=1)
    print("Hyperparameters...")
    for column in hyperparams:
        name = column.split("/")[1]
        value = hyperparams[column].values[0]
        print(f"- {name}: {value}")

    return analysis.get_best_config()


def save_model(model, config, path_weights, path_config):
    """Save trained torch weights with config"""
    torch.save(model.state_dict(), path_weights)

    with open(path_config, "w") as fp:
        json.dump(config, fp, indent=4)


def predict_screen_errors(
    *img_types,
    path_model,
    path_config,
    path_plot=None,
    path_errors=None,
    data_partial=True,
    steps=10,
):
    """Get prediction error for each screen coordinate"""
    with open(path_config) as json_file:
        config = json.load(json_file)

    if len(img_types) == 1:
        model = SingleModel(config, img_types[0])
    else:
        model = FullModel(config)

    model.load_state_dict(torch.load(path_model))
    model.cuda()
    model.eval()

    data = FaceDataset(Path.cwd(), data_partial, *img_types)

    x = []
    y = []
    error = []

    for i, d in tqdm(enumerate(data), total=len(data)):
        if i % steps == 0:
            img_list = [d[img].unsqueeze(0).cuda() for img in img_types]

            with torch.no_grad():
                target = d["targets"].cuda()
                predict = model(*img_list)[0]
                dist = torch.sqrt(((predict - target) ** 2).sum(axis=0))

                x.append(target.cpu().numpy()[0])
                y.append(target.cpu().numpy()[1])
                error.append(float(dist.cpu().numpy()))

    print(
        "Average error: {}px over {} predictions".format(
            round(np.mean(error), 2), len(error)
        )
    )
    errors = plot_screen_errors(
        x,
        y,
        error,
        path_plot=path_plot,
        path_errors=path_errors,
    )

    return errors


def plot_screen_errors(x, y, z, path_plot=None, path_errors=None):
    """Plot prediction errors over screen space"""
    # create grid
    xi = np.arange(0, 1920, 1)
    yi = np.arange(0, 1080, 1)
    xi, yi = np.meshgrid(xi, yi)

    # interpolate
    zi = griddata((x, y), z, (xi, yi), method="nearest")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.contourf(xi, yi, zi)
    cbar = plt.colorbar()

    cbar.ax.set_ylabel("Error (px)")
    plt.xlabel("Screen X")
    plt.ylabel("Screen Y")
    plt.gca().invert_yaxis()

    if path_plot is not None:
        plt.savefig(path_plot, dpi=100)

    if path_errors is not None:
        np.save(path_errors, zi.T)

    plt.show()

    # Error histogram
    plt.hist(z, edgecolor="black")
    plt.xlabel("Error (px)")
    plt.ylabel("Count")
    plt.show()

    return zi.T


# Tensorflow test things below


class OrderedDictWithDefaultList(OrderedDict):
    """
    Used for tensorflow in-memory datasets
    """

    def __missing__(self, key):
        value = list()
        self[key] = value
        return value


def create_data_splits(data, train_size=0.80, shuffle=True, random_state=87):
    """
    Used for tensorflow in-memory datasets

    This mutates the data dict.
    To make it less brittle (at the cost of more memory), create and return a data.copy()
    """
    # Create training set
    split1 = train_test_split(
        *[v for v in data.values()],
        test_size=1 - train_size,
        shuffle=shuffle,
        random_state=random_state,
    )
    train_data = split1[0::2]
    remaining_data = split1[1::2]

    # Split remaining into validation and test sets (50/50)
    split2 = train_test_split(
        *remaining_data,
        test_size=0.5,
        shuffle=shuffle,
        random_state=random_state,
    )
    val_data = split2[0::2]
    test_data = split2[1::2]

    for i, k in enumerate(data.keys()):
        data[k] = {}
        data[k]["train"] = train_data[i]
        data[k]["val"] = val_data[i]
        data[k]["test"] = test_data[i]

    return data
