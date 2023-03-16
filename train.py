import os
from pathlib import Path

import colossalai
import torch
import torchvision
import torchvision.transforms as transforms
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from colossalai.nn.metric import Accuracy
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from config.argment import My_parser
from models.alexnet import AlexNet
from src.utils import setup_os_parameter_and_seed

DATA_ROOT = Path(os.environ.get("CIFAR10", "./data"))


def main():
    parser = My_parser()
    parser.add_argument(
        "--use_trainer",
        action="store_true",
        default=True,
        help="whether to use trainer",
    )
    args = parser.parse_args()

    setup_os_parameter_and_seed(args, seed=123)

    colossalai.launch_from_torch(config=args.config)

    logger = get_dist_logger()

    model = AlexNet(gpc.config.NUM_CLASSES)

    train_transform = transforms.Compose(
        [
            # transforms.RandomCrop(size=32, padding=4),
            transforms.Resize(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(size=224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )

    train_dataset = CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=train_transform
    )
    test_dataset = CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=test_transform
    )

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
    )
    test_dataloader = get_dataloader(
        dataset=train_dataset,
        add_sampler=False,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
    )

    # build criterion
    criterion = CrossEntropyLoss()
    # optimizer
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # lr_scheduler
    lr_scheduler = CosineAnnealingLR(
        optimizer=optimizer, total_steps=gpc.config.NUM_EPOCHS
    )

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model, optimizer, criterion, train_dataloader, test_dataloader
    )

    if not args.use_trainer:
        for epoch in range(gpc.config.NUM_EPOCHS):
            engine.train()
            if gpc.get_global_rank() == 0:
                train_dl = tqdm(train_dataloader)
            else:
                train_dl = train_dataloader
            for img, label in train_dl:
                img = img.cuda()
                label = label.cuda()

                engine.zero_grad()
                output = engine(img)
                train_loss = engine.criterion(output, label)
                engine.backward(train_loss)
                engine.step()
            lr_scheduler.step()

            engine.eval()
            correct = 0
            total = 0

            for img, label in test_dataloader:
                img = img.cuda()
                label = label.cuda()

                with torch.no_grad():
                    output = engine(img)
                    test_loss = engine.criterion(output, label)
                pred = torch.argmax(output, dim=-1)
                correct += torch.sum(pred == label)
                total += img.size(0)

            logger.info(
                f"Epoch {epoch} - train loss: {train_loss:.5}, test loss: {test_loss:.5}, acc: {correct / total:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}",
                ranks=[0],
            )
    else:
        # build a timer to measure time
        timer = MultiTimer()

        # create a trainer object
        trainer = Trainer(engine=engine, timer=timer, logger=logger)

        # definer the hooks to attach to the trainer
        hook_list = [
            hooks.LossHook(),
            hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
            hooks.AccuracyHook(accuracy_func=Accuracy()),
            hooks.LogMetricByEpochHook(logger),
            hooks.LogMemoryByEpochHook(logger),
            hooks.LogTimingByEpochHook(timer, logger),
            # you can uncomment these lines if you wish to use them
            hooks.TensorboardHook(log_dir="./data/tb_logs", ranks=[0]),
            hooks.SaveCheckpointHook(checkpoint_dir="./data/ckpt"),
        ]

        # start training
        trainer.fit(
            train_dataloader=train_dataloader,
            epochs=gpc.config.NUM_EPOCHS,
            test_dataloader=test_dataloader,
            test_interval=1,
            hooks=hook_list,
            display_progress=True,
        )


if __name__ == "__main__":
    main()
