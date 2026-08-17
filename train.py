import torch
import torch.nn as nn
import logging
import argparse
import torch.optim as optim
import time
import logging
import math
import os
from dataset import RawFolder, SonyFolder, SamsungFolder, OlympusFolder, NUSFolder, SonyTestFolder
import random
from torch.utils.data import DataLoader
import sys
from tensorboardX import SummaryWriter
from models import RawCompressor_color_scale_bit
import pdb

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, out_compress, shape):
        out = {}
        yz_likelihoods = out_compress['yz_likelihoods']
        x_likelihoods = out_compress['x_likelihoods']
        B, C, H, W = shape
        num_pixels = B * C * H * W
        
        out["yz_bpp"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in yz_likelihoods.values()
        )
        
        out["x_bpp"] = x_likelihoods.sum() / num_pixels
        out["loss"] = out["x_bpp"] + out["yz_bpp"]
        return out


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model,
    criterion,
    train_dataloader,
    optimizer,
    aux_optimizer,
    epoch,
    clip_max_norm,
):
    model.train()
    device = next(model.parameters()).device
    bpp_loss = AverageMeter()
    yz_loss = AverageMeter()
    x_loss = AverageMeter()
    with torch.autograd.set_detect_anomaly(True):
        for i, d in enumerate(train_dataloader):
            data, black_level, white_level = d
            data = data.to(device)
            black_level = black_level.to(device)
            white_level = white_level.to(device)
 
            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            out_net = model(data, black_level, white_level)

            out_criterion = criterion(out_net, data.shape)
            out_criterion["loss"].backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
            bpp_loss.update(out_criterion['loss'].item())
            yz_loss.update(out_criterion['yz_bpp'].item())
            x_loss.update(out_criterion['x_bpp'].item())
        

            if i % 10 == 0:
                logging.info(
                    f"Train epoch {epoch}: ["
                    f"{i * len(data)}/{len(train_dataloader.dataset)}"
                    f" ({100.0 * i / len(train_dataloader):.0f}%)]"
                    f"\tLoss: {out_criterion['loss'].item():.3f} |"
                    f"\tyz loss: {out_criterion['yz_bpp'].item():.3f} |"
                    f"\tBpp loss: {out_criterion['x_bpp'].item():.2f} |"
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
    return bpp_loss.avg, yz_loss.avg, x_loss.avg


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device
    bpp_loss = AverageMeter()
    yz_loss = AverageMeter()
    x_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            data, black_level, white_level = d
            data = data.to(device)  
            black_level = black_level.to(device)
            white_level = white_level.to(device)
            out_net = model(data,black_level,white_level)
            out_criterion = criterion(out_net, data.shape)
            bpp_loss.update(out_criterion["loss"])
            yz_loss.update(out_criterion["yz_bpp"])
            x_loss.update(out_criterion['x_bpp'])
                
        logging.info(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {bpp_loss.avg:.3f} |"
            f"\tyz loss: {yz_loss.avg:.3f} |"
            f"\tx loss: {x_loss.avg:.3f} \n"
        )

    return bpp_loss.avg, yz_loss.avg, x_loss.avg


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=256,
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--save_path", type=str, help="save_path")
    parser.add_argument("--skip_epoch", type=int, default=0)
    parser.add_argument(
        "--N",
        type=int,
        default=128,
    )
    parser.add_argument("--lr_epoch", nargs="+", type=int)
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--all", action='store_true')
    parser.add_argument("--sony", action='store_true')
    parser.add_argument("--samsung", action='store_true')
    parser.add_argument("--olympus", action='store_true')
    parser.add_argument("--nikon", action='store_true')
    args = parser.parse_args(argv)
    return args


def save_checkpoint(state, is_best, epoch, save_path, filename):
    torch.save(state, save_path + "/checkpoint_latest.pth.tar")
    if is_best:
        torch.save(state, save_path + "/checkpoint_best.pth.tar")


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
        logging.info(arg, ":", getattr(args, arg))
    save_path = args.save_path
    setup_logger(save_path + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "/tensorboard/")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    writer = SummaryWriter(save_path + "/tensorboard/")
    
    if args.nikon:
        train_dataset = RawFolder(root=args.dataset, split="train", crop='random', patch_size=args.patch_size, bit_depth=14)
        test_dataset = RawFolder(root=args.dataset, split="test", crop='center', patch_size=512, bit_depth=14)
    
    if args.sony:
        train_dataset = SonyFolder(root=args.dataset, split="train_patch", crop='random', patch_size=args.patch_size)
        test_dataset = SonyTestFolder(root=args.dataset, split="test", crop='center', patch_size=512)
    
    if args.samsung:
        train_dataset = SamsungFolder(root=args.dataset, split="train_patch", crop='random', patch_size=args.patch_size)
        test_dataset = NUSFolder(root=args.dataset, split="test", crop='center', patch_size=512)
    
    if args.olympus:
        train_dataset = OlympusFolder(root=args.dataset, split="train_patch", crop='random', patch_size=args.patch_size)
        test_dataset = NUSFolder(root=args.dataset, split="test", crop='center', patch_size=512)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)
    logging.info(device)
    device = "cuda"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    
    bit = 14 if args.nikon else 12
    if args.all:
        net = RawCompressor_color_scale_bit(bit=bit)
    
    net = net.to(device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=0.1, last_epoch=-1
    )

    criterion = Loss()

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        logging.info("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss, train_yz_loss, train_x_loss = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,            
        )
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_yz_loss", train_loss, epoch)
        writer.add_scalar("train_x_loss", train_x_loss, epoch)
        
        loss, yz_loss, x_loss = test_epoch(epoch, test_dataloader, net, criterion)
        writer.add_scalar("test_loss", loss, epoch)
        writer.add_scalar("test_yz_loss", yz_loss, epoch)
        writer.add_scalar("test_x_loss", x_loss, epoch)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                epoch,
                save_path,
                save_path + str(epoch) + "_checkpoint.pth.tar",
            )


if __name__ == "__main__":
    main(sys.argv[1:])
