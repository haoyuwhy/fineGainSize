import argparse
import os
import sys
import os.path as osp

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim
from tqdm import tqdm
from defaults import get_default_cfg
from my_dataset import MyDataSet
from utils.utils import read_split_data,mkdir

from models.model import build_resnet

def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(cfg.INPUT.DATA_ROOT)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    val_data_set = MyDataSet(images_path=val_images_path,
                               images_class=val_images_label,
                               transform=data_transform["val"])

    batch_size = cfg.INPUT.BATCH_SIZE_TRAIN
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)
    validate_loader = torch.utils.data.DataLoader(val_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               collate_fn=val_data_set.collate_fn)

    # plot_data_loader_image(train_loader)
    # print(len(val_data_set))#164991
    print("Creating model")
    net = build_resnet(cfg.BACKBONE, cfg.BACKBONE_PRETRAINED)
    # net = resnet34()
    # # load pretrain weights
    # # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = "./resnet34-333f7ec4.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 5)
    net.fc = nn.Linear(in_channel, len(list(set(train_images_label))))
    net.to(device)
    print(net)

    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    tb_writer = SummaryWriter(log_dir=osp.join(cfg.OUTPUT_DIR, "tf_log_acc"))

    epochs = cfg.INPUT.EPOCH
    # best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / len(val_data_set)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        tb_writer.add_scalar('acc', val_accurate, epoch)
        tb_writer.add_scalar('running_loss', running_loss, epoch)
        tb_writer.add_scalar('running_loss/train_steps', running_loss / train_steps, epoch)

        torch.save(net.state_dict(), osp.join(cfg.OUTPUT_DIR, f"epoch_{epoch}.pth"))
        # if val_accurate > best_acc:
        #     best_acc = val_accurate
        #     torch.save(net.state_dict(), osp.join(cfg.OUTPUT_DIR, f"best_epoch_{epoch}.pth"))

    print('Finished Training')
    tb_writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a finegainsize network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the specified checkpoint."
    )
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    args = parser.parse_args()
    main(args)
