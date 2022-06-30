from torch import nn
import torch
import pytorch_lightning as pl
from collections import OrderedDict
import torchmetrics


class WindNet(nn.Module):
    def __init__(self, args) -> None:
        super(WindNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=args["in_channels"],
            out_channels=args["out_channels_1"],  # 12
            kernel_size=args["k_size_1"],
            stride=args["stride_1"],
            dilation=args["dilation_1"],
            padding=args["k_size_1"] - 1,
        )
        self.conv2 = nn.Conv2d(  # 12, 6, 3
            in_channels=args["out_channels_1"],
            out_channels=args["out_channels_2"],
            kernel_size=args["k_size_2"],
            stride=args["stride_2"],
            dilation=args["dilation_2"],
            padding=args["k_size_2"] - 1,
        )
        self.maxpool = nn.MaxPool2d(args["maxpool_2"])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(args["fc_size"], 2)
        self.args = args
        self.net = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.maxpool,
            self.flatten,
            self.fc,
        ).double()

    def forward(self, X) -> torch.Tensor:
        output = self.net(X)
        return output


class WindNetPL(pl.LightningModule):
    ## Initialize. Define latent dim, learning rate, and Adam betas
    def __init__(self, args):
        super().__init__()
        # self.save_hyperparameters()
        self.args = args
        self.net = WindNet(self.args)

        self.accuracy = torchmetrics.Accuracy()
        self.AUROC = torchmetrics.AUROC(num_classes=2)
        self.precision_m = torchmetrics.Precision()
        self.recall = torchmetrics.Recall()
        # self.conf_matrix = torchmetrics.ConfusionMatrix(num_classes=2)

        self.loss_f = nn.BCEWithLogitsLoss(pos_weight=self.args["pos_weight"])

    def forward(self, X):
        return self.net(X)

    def loss(
        self, y_hat, y
    ):  # POS WEIGHT CLASS !!! https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        return self.loss_f(y_hat, y)

    def training_step(self, batch, batch_idx):
        objs, target = batch

        predictions = self(objs)
        loss = self.loss(predictions, target)

        # logging

        self.log("train_loss", loss, prog_bar=True)

        tqdm_dict = {
            "train_loss": loss,
        }
        output = OrderedDict(
            {
                "loss": loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
                "preds": predictions,
                "target": target,
            }
        )
        return output

    def training_step_end(self, outputs):
        # update and log
        predictions = outputs["preds"]
        target = outputs["target"]
        self.accuracy(predictions, target.argmax(dim=1))
        self.recall(predictions, target.argmax(dim=1))
        self.AUROC(predictions, target.argmax(dim=1))
        self.precision_m(predictions, target.argmax(dim=1))
        # self.conf_matrix(predictions, target.argmax(dim=1))
        self.log("train_acc_step", self.accuracy, prog_bar=True)
        self.log("train_recall_step", self.recall, prog_bar=True)
        self.log("train_AUROC_step", self.AUROC, prog_bar=True)
        self.log("train_precision_step", self.precision_m, prog_bar=True)
        # self.log("train_conf_matrix_step", self.conf_matrix)

    #     # self.metric(outputs['preds'], outputs['target'])
    #     self.log('metric', self.metric)

    def validation_step(self, batch, batch_idx):
        objs, target = batch

        predictions = self(objs)
        loss = self.loss(predictions, target)

        # logging
        # self.logger.experiment.add_image("generated_images", grid, 0)

        self.log("val_loss", loss, prog_bar=True)
        tqdm_dict = {"val_loss": loss}
        output = OrderedDict(
            {
                "loss": loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
                "preds": predictions,
                "target": target,
            }
        )
        return output

    def validation_step_end(self, outputs):
        # update and log
        predictions = outputs["preds"]
        target = outputs["target"]
        acc = self.accuracy(predictions, target.argmax(dim=1))
        rec = self.recall(predictions, target.argmax(dim=1))
        auroc = self.AUROC(predictions, target.argmax(dim=1))
        prec = self.precision_m(predictions, target.argmax(dim=1))
        self.logger.experiment.add_scalar("val_acc", acc)
        self.logger.experiment.add_scalar("val_recall", rec)
        self.logger.experiment.add_scalar("val_AUROC", auroc)
        self.logger.experiment.add_scalar("val_precision", prec)
        # self.conf_matrix(predictions, target.argmax(dim=1))
        self.log("val_acc_step", self.accuracy, prog_bar=True)
        self.log("val_recall_step", self.recall, prog_bar=True)
        self.log("val_AUROC_step", self.AUROC, prog_bar=True)
        self.log("val_precision_step", self.precision_m, prog_bar=True)

    def test_step(self, batch, batch_idx):
        objs, target = batch

        predictions = self(objs)
        loss = self.loss(predictions, target)

        # logging
        # self.logger.experiment.add_image("generated_images", grid, 0)

        self.log("val_loss", loss, prog_bar=True)
        tqdm_dict = {"val_loss": loss}
        output = OrderedDict(
            {
                "loss": loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
                "preds": predictions,
                "target": target,
            }
        )
        return output

    def test_step_end(self, outputs):
        # update and log
        predictions = outputs["preds"]
        target = outputs["target"]
        self.accuracy(predictions, target.argmax(dim=1))
        self.recall(predictions, target.argmax(dim=1))
        self.AUROC(predictions, target.argmax(dim=1))
        self.precision_m(predictions, target.argmax(dim=1))
        # self.conf_matrix(predictions, target.argmax(dim=1))
        self.log("test_acc_step", self.accuracy, prog_bar=True)
        self.log("test_recall_step", self.recall, prog_bar=True)
        self.log("test_AUROC_step", self.AUROC, prog_bar=True)
        self.log("test_precision_step", self.precision_m, prog_bar=True)

    def configure_optimizers(self):
        lr = self.args["lr"]
        b1 = self.args["b1"]
        b2 = self.args["b2"]

        opt = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(b1, b2))
        return [opt], []
