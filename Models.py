import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from collections import OrderedDict
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


def create_datasets(
    cwd, data_partial, img_types, batch_size=1, train_prop=0.8, val_prop=0.1, seed=87
):
    dataset = FaceDataset(cwd, data_partial, *img_types)
    n_train = int(len(dataset) * train_prop)
    n_val = int(len(dataset) * val_prop)
    n_test = len(dataset) - n_train - n_val
    ds_train, ds_val, ds_test = random_split(
        dataset, (n_train, n_val, n_test), generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    test_loader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    return train_loader, val_loader, test_loader


class FaceDataset(Dataset):
    def __init__(self, cwd, data_partial, *img_types):
        if data_partial:
            self.dir_data = cwd / "data_partial"
        else:
            self.dir_data = cwd / "data"

        df = pd.read_csv(self.dir_data / "positions.csv")
        df["filename"] = df["id"].astype("str") + ".jpg"

        self.img_types = list(img_types)
        self.filenames = df["filename"].tolist()
        self.targets = torch.Tensor(list(zip(df["x"], df["y"])))
        self.head_angle = torch.Tensor(df["head_angle"].tolist())

        self.transform = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        batch = {
            "targets": self.targets[idx],
        }

        if "head_angle" in self.img_types:
            batch["head_angle"] = self.head_angle[idx]

        for img_type in self.img_types:
            if not img_type == "head_angle":
                img = Image.open(
                    self.dir_data / f"{img_type}" / f"{self.filenames[idx]}"
                )
                if img_type == "head_pos":
                    # Head pos is a generated black/white image and shouldn't be adjusted
                    img = transforms.ToTensor()(img)
                else:
                    img = self.transform(img)
                batch[img_type] = img

        return batch


class SingleModel(pl.LightningModule):
    def __init__(self, config, img_type):
        super().__init__()
        self.save_hyperparameters()  # stores hparams in saved checkpoint files

        feat_size = 64
        self.example_input_array = torch.rand(1, 3, feat_size, feat_size)

        self.img_type = img_type
        self.lr = config["lr"]
        self.filter_size = config["filter_size"]
        self.filter_growth = config["filter_growth"]
        self.n_filters = config["n_filters"]
        self.n_convs = config["n_convs"]
        self.dense_nodes = config["dense_nodes"]

        # First layer after input
        self.conv_input = nn.Conv2d(3, self.n_filters, self.filter_size)
        feat_size = feat_size - (self.filter_size - 1)

        # Additional conv layers
        self.convs1 = nn.ModuleList()
        n_out = self.n_filters
        for i in range(self.n_convs):
            n_in = n_out
            n_out = n_in * self.filter_growth

            self.convs1.append(self.conv_block(n_in, n_out, self.filter_size))
            # Calculate input feature size reductions due to conv and pooling
            feat_size = (feat_size - (self.filter_size - 1)) // 2

        # FC layers -> output
        self.drop1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(n_out * feat_size * feat_size, self.dense_nodes)
        self.drop2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(self.dense_nodes, self.dense_nodes // 2)
        self.fc3 = nn.Linear(self.dense_nodes // 2, 2)

    def forward(self, x):
        x = self.conv_input(x)

        for c in self.convs1:
            x = c(x)

        x = x.reshape(x.shape[0], -1)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def conv_block(self, input_size, output_size, filter_size):
        block = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(input_size, output_size, filter_size)),
                    ("relu", nn.ReLU()),
                    ("norm", nn.BatchNorm2d(output_size)),
                    ("pool", nn.MaxPool2d((2, 2))),
                ]
            )
        )
        return block

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch[self.img_type], batch["targets"]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[self.img_type], batch["targets"]
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch[self.img_type], batch["targets"]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)
        return loss


class EyesModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()  # stores hparams in saved checkpoint files

        feat_size = 64
        self.example_input_array = [torch.rand(1, 3, feat_size, feat_size)] * 2

        self.lr = config["lr"]
        self.filter_size = config["filter_size"]
        self.filter_growth = config["filter_growth"]
        self.n_filters = config["n_filters"]
        self.n_convs = config["n_convs"]
        self.dense_nodes = config["dense_nodes"]

        # Left eye
        # First layer after input
        self.l_conv_input = nn.Conv2d(3, self.n_filters, self.filter_size)
        self.r_conv_input = nn.Conv2d(3, self.n_filters, self.filter_size)
        feat_size = feat_size - (self.filter_size - 1)

        # Additional conv layers
        self.l_convs = nn.ModuleList()
        self.r_convs = nn.ModuleList()

        n_out = self.n_filters
        for _ in range(self.n_convs):
            n_in = n_out
            n_out = n_in * self.filter_growth

            self.l_convs.append(self.conv_block(n_in, n_out, self.filter_size, "l"))
            self.r_convs.append(self.conv_block(n_in, n_out, self.filter_size, "r"))

            # Calculate input feature size reductions due to conv and pooling
            feat_size = (feat_size - (self.filter_size - 1)) // 2

        # FC layers -> output
        self.drop1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(n_out * feat_size * feat_size * 2, self.dense_nodes)
        self.drop2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(self.dense_nodes, self.dense_nodes // 2)
        self.fc3 = nn.Linear(self.dense_nodes // 2, 2)

    def forward(self, l_eye, r_eye):
        l_eye = self.l_conv_input(l_eye)
        for c in self.l_convs:
            l_eye = c(l_eye)

        r_eye = self.r_conv_input(r_eye)
        for c in self.r_convs:
            r_eye = c(r_eye)

        out = torch.cat((l_eye, r_eye), dim=1)

        out = out.reshape(out.shape[0], -1)
        out = self.drop1(F.relu(self.fc1(out)))
        out = self.drop2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out

    def conv_block(self, input_size, output_size, filter_size, name):
        block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "{}_conv".format(name),
                        nn.Conv2d(input_size, output_size, filter_size),
                    ),
                    ("{}_relu".format(name), nn.ReLU()),
                    ("{}_norm".format(name), nn.BatchNorm2d(output_size)),
                    ("{}_pool".format(name), nn.MaxPool2d((2, 2))),
                ]
            )
        )
        return block

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        l_eye, r_eye = batch["l_eye"], batch["r_eye"]
        y_hat = self(l_eye, r_eye)
        loss = F.mse_loss(y_hat, batch["targets"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        l_eye, r_eye = batch["l_eye"], batch["r_eye"]
        y_hat = self(l_eye, r_eye)
        val_loss = F.mse_loss(y_hat, batch["targets"])
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        l_eye, r_eye = batch["l_eye"], batch["r_eye"]
        y_hat = self(l_eye, r_eye)
        loss = F.mse_loss(y_hat, batch["targets"])
        self.log("test_loss", loss)
        return loss


class FullModel(pl.LightningModule):
    def __init__(self, config):
        """
        I really really really wish pytorch had a way to calculate layer output sizes during init...
        """
        super().__init__()
        self.save_hyperparameters()  # stores hparams in saved checkpoint files

        feat_size = 64
        self.lr = config["lr"]

        # Example input for graph logging
        graph_example = [torch.rand(1, 3, feat_size, feat_size)] * 3
        graph_example.append(torch.rand(1, 1, feat_size, feat_size))
        graph_example.append(torch.rand(1))
        self.example_input_array = graph_example

        # Face input
        self.face_conv_input = nn.Conv2d(
            3, config["n_face_filt"], config["face_filt_size"]
        )
        face_feat_size = feat_size - (config["face_filt_size"] - 1)

        self.face_convs = nn.ModuleList()
        n_out_face = config["n_face_filt"]
        for _ in range(config["n_face_conv"]):
            n_in_face = n_out_face
            n_out_face = n_in_face * config["face_filt_grow"]

            self.face_convs.append(
                self.conv_block(n_in_face, n_out_face, config["face_filt_size"], "face")
            )

            # Calculate input feature size reductions due to conv and pooling
            face_feat_size = (face_feat_size - (config["face_filt_size"] - 1)) // 2

        face_feat_shape = (
            n_out_face,
            face_feat_size,
            face_feat_size,
        )
        face_feat_len = math.prod(face_feat_shape)

        # Eye inputs
        self.l_conv_input = nn.Conv2d(3, config["n_eye_filt"], config["eye_filt_size"])
        self.l_convs = nn.ModuleList()

        self.r_conv_input = nn.Conv2d(3, config["n_eye_filt"], config["eye_filt_size"])
        self.r_convs = nn.ModuleList()

        eye_feat_size = feat_size - (config["eye_filt_size"] - 1)

        n_out_eye = config["n_eye_filt"]
        for _ in range(config["n_eye_conv"]):
            n_in_eye = n_out_eye
            n_out_eye = n_in_eye * config["eye_filt_grow"]

            self.l_convs.append(
                self.conv_block(n_in_eye, n_out_eye, config["eye_filt_size"], "l")
            )
            self.r_convs.append(
                self.conv_block(n_in_eye, n_out_eye, config["eye_filt_size"], "r")
            )

            # Calculate input feature size reductions due to conv and pooling
            eye_feat_size = (eye_feat_size - (config["eye_filt_size"] - 1)) // 2

        eye_feat_shape = (
            n_out_eye,
            eye_feat_size,
            eye_feat_size,
        )
        eye_feat_len = math.prod(eye_feat_shape)

        # Head pos input
        self.head_pos_conv_input = nn.Conv2d(
            1, config["n_head_pos_filt"], config["head_pos_filt_size"]
        )
        head_pos_feat_size = feat_size - (config["head_pos_filt_size"] - 1)

        self.head_pos_convs = nn.ModuleList()
        n_out_head_pos = config["n_head_pos_filt"]
        for _ in range(config["n_head_pos_conv"]):
            n_in_head_pos = n_out_head_pos
            n_out_head_pos = n_in_head_pos * config["head_pos_filt_grow"]

            self.head_pos_convs.append(
                self.conv_block(
                    n_in_head_pos,
                    n_out_head_pos,
                    config["head_pos_filt_size"],
                    "head_pos",
                )
            )

            # Calculate input feature size reductions due to conv and pooling
            head_pos_feat_size = (
                head_pos_feat_size - (config["head_pos_filt_size"] - 1)
            ) // 2

        head_pos_feat_shape = (
            n_out_head_pos,
            head_pos_feat_size,
            head_pos_feat_size,
        )
        head_pos_feat_len = math.prod(head_pos_feat_shape)

        # FC layers -> output
        self.drop1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(
            face_feat_len + eye_feat_len * 2 + head_pos_feat_len + 1,
            config["dense_nodes"],
        )
        self.drop2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(config["dense_nodes"], config["dense_nodes"] // 2)
        self.fc3 = nn.Linear(config["dense_nodes"] // 2, 2)

    def forward(self, face, l_eye, r_eye, head_pos, head_angle):
        face = self.face_conv_input(face)
        for c in self.face_convs:
            face = c(face)
        face = face.flatten(start_dim=1)

        l_eye = self.l_conv_input(l_eye)
        for c in self.l_convs:
            l_eye = c(l_eye)
        l_eye = l_eye.flatten(start_dim=1)

        r_eye = self.r_conv_input(r_eye)
        for c in self.r_convs:
            r_eye = c(r_eye)
        r_eye = r_eye.flatten(start_dim=1)

        head_pos = self.head_pos_conv_input(head_pos)
        for c in self.head_pos_convs:
            head_pos = c(head_pos)
        head_pos = head_pos.flatten(start_dim=1)

        # Combine conv outputs, add head angle
        out = torch.hstack([face, l_eye, r_eye, head_pos])
        out = torch.hstack([out, head_angle.unsqueeze(1)])

        out = self.drop1(F.relu(self.fc1(out)))
        out = self.drop2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out

    def conv_block(self, input_size, output_size, filter_size, name):
        block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "{}_conv".format(name),
                        nn.Conv2d(input_size, output_size, filter_size),
                    ),
                    ("{}_relu".format(name), nn.ReLU()),
                    ("{}_norm".format(name), nn.BatchNorm2d(output_size)),
                    ("{}_pool".format(name), nn.MaxPool2d((2, 2))),
                ]
            )
        )
        return block

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        face_aligned, l_eye, r_eye, head_pos, head_angle = (
            batch["face_aligned"],
            batch["l_eye"],
            batch["r_eye"],
            batch["head_pos"],
            batch["head_angle"],
        )
        y_hat = self(face_aligned, l_eye, r_eye, head_pos, head_angle)
        loss = F.mse_loss(y_hat, batch["targets"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        face_aligned, l_eye, r_eye, head_pos, head_angle = (
            batch["face_aligned"],
            batch["l_eye"],
            batch["r_eye"],
            batch["head_pos"],
            batch["head_angle"],
        )
        y_hat = self(face_aligned, l_eye, r_eye, head_pos, head_angle)
        val_loss = F.mse_loss(y_hat, batch["targets"])
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        face_aligned, l_eye, r_eye, head_pos, head_angle = (
            batch["face_aligned"],
            batch["l_eye"],
            batch["r_eye"],
            batch["head_pos"],
            batch["head_angle"],
        )
        y_hat = self(face_aligned, l_eye, r_eye, head_pos, head_angle)
        loss = F.mse_loss(y_hat, batch["targets"])
        self.log("test_loss", loss)
        return loss