from torch.optim.lr_scheduler import StepLR
from models import CCT_Model
import torch, argparse, os, time, warnings, sys
from utils.engine import train_step, val_step
from utils.datasets import build_dataset
from torch.utils.data import DataLoader
from utils.estimate_model import Predictor, Plot_ROC
from utils.scheduler import create_lr_scheduler


warnings.filterwarnings('ignore')

def main(args):

    print(args)

    if os.path.exists("./CCT_Save_weights") is False:
        os.makedirs("./CCT_Save_weights")

    train_dataset, val_dataset = build_dataset(args)

    batch_size = args.batch_size

    sys_name = sys.platform
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]) if 'linux' in sys_name.lower() else 0  # number workers
    print('Using {} dataloader workers every process'.format(nw))


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_dataset.collate_fn)


    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)


    # model
    model = CCT_Model(num_classes=args.num_classes)
    model = model.to(args.device)

    best_score = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.33)

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    scalar = torch.cuda.amp.GradScaler() if args.scalar else None

    resume = os.path.exists(f'./CCT_Save_weights/model.pth')

    if resume:
        path_checkpoint = f"./CCT_Save_weights/model.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['model'])  # 加载模型可学习参数
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        start_epoch = checkpoint['epoch']
        best_score = checkpoint['best_score']

        if scalar:
            scalar.load_state_dict(checkpoint["scalar"])

        print(f'from {start_epoch} epoch starts training!!!')

    print('Start Training!!!')

    start_time = time.time()

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_step(net=model,
                                           optimizer=optimizer,
                                           lr_scheduler=lr_scheduler,
                                           data_loader=train_loader,
                                           device=args.device,
                                           epoch=epoch,
                                           scalar=scalar)


        # validate
        val_loss, val_acc = val_step(net=model,
                                     data_loader=val_loader,
                                     device=args.device,
                                     epoch=epoch)


        if val_acc > best_score:

            best_score = val_acc

            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "best_score": best_score,
                         "args": args}
            if scalar:
                save_file["scalar"] = scalar.state_dict()
            torch.save(save_file, "./CCT_Save_weights/model.pth")

    end_time = time.time()
    print(f'Use {round(end_time - start_time, 3)}seconds')
    print('Finished Training!!!')

    # predict and plot ROC
    print('*******************STARTING PREDICT*******************')
    Predictor(model, val_loader, args.resume, args.device)
    Plot_ROC(model, val_loader, args.resume, args.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5) # number classes
    parser.add_argument('--epochs', type=int, default=5) # train epochs
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001) # learning_rate
    parser.add_argument('--weight-decay', type=float, default=1e-5)  # weight_decay

    # data directory
    parser.add_argument('--data-path', type=str, default=r"./flower_data")

    # pretrain weights directory
    # parser.add_argument('--weights', type=str, default='', help='initial weights path')

    # chill weights
    # parser.add_argument('--freeze-layers', type=bool, default=False)

    # checkpoint
    # parser.add_argument('--resume', type=bool, default=True)

    # amp
    parser.add_argument('--scalar', type=bool, default=True)
    # device
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)
