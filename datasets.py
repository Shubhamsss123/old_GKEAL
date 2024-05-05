import numpy as np
import torch
from torchvision import datasets, transforms
from main import args
import random
import copy
from uni_sampling import small_batch_dataloader
from torch.utils.data import TensorDataset
if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class KuzushijiMNIST(datasets.MNIST):
    urls = [
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz'
    ]


kwargs = {'num_workers': args.workers, 'pin_memory': True}

if args.dataset == 'cifar100':
    input_dim = 32
    input_ch = 3
    num_classes = 100

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    '''
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    '''

    dataset_train = datasets.CIFAR100('../../data/cifar100', train=True, download=True, transform=train_transform)
    dataset_val = datasets.CIFAR100('../../data/cifar100', train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                 std=[0.267, 0.256, 0.276])
                                        ]))
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(dataset_val,
                                             batch_size=args.batch_size,
                                             shuffle=False, **kwargs)


elif args.dataset == 'tiny-imagenet':
    input_dim = 64
    input_ch = 3
    num_classes = 200
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    ])
    dataset_train = datasets.ImageFolder(args.data+'/train', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs
                                               )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data+'/val',
                             transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
                             ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

elif args.dataset == 'imagenet100':
    input_dim = 224
    input_ch = 3
    num_classes = 100
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset_train = datasets.ImageFolder('../train_folder', transform=train_transform)
    labels = np.array([a[1] for a in dataset_train.samples])
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)

    dataset_val = datasets.ImageFolder('../test_folder',
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                             ]))
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size, shuffle=False, **kwargs)

elif args.dataset == 'cub200':
    input_dim = 224
    input_ch = 3
    num_classes = 200

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset_train = datasets.ImageFolder('../train_folder', transform=train_transform)
    labels = np.array([a[1] for a in dataset_train.samples])
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)

    dataset_val = datasets.ImageFolder('../test_folder',
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                             ]))
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size, shuffle=False, **kwargs)


elif args.dataset == 'imagenet':
    input_dim = 224
    input_ch = 3
    num_classes = 1000
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset_train = datasets.ImageFolder(args.data+'/train', transform=train_transform)
    labels = np.array([a[1] for a in dataset_train.samples])
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)

    dataset_val = datasets.ImageFolder(args.data+'/val2',
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                             ]))
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size, shuffle=False, **kwargs)
else:
    print('No valid dataset is specified')

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
else:
    train_sampler = None

def get_IL_dataset(orginal_loader, IL_loader, shuffle):
    orginal_dataset = orginal_loader.dataset
    IL_dataset = IL_loader.dataset
    com_dataset = torch.utils.data.ConcatDataset([orginal_dataset, IL_dataset])
    com_loader = torch.utils.data.DataLoader(dataset=com_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=shuffle, **kwargs)
    return com_loader


def get_IL_loader(dataset_train, dataset_val, n_classes):
    if args.dataset in ['imagenet', 'imagenet100']:
        targets_train = torch.tensor(np.array([a[1] for a in dataset_train.samples]))
        targets_val = torch.tensor(np.array([a[1] for a in dataset_val.samples]))
    else:
        targets_train = torch.tensor(dataset_train.targets)
        targets_val = torch.tensor(dataset_val.targets)
    for idx in range(args.baseclass):
        if idx == 0:
            target_idx_train = (targets_train == 0).nonzero()
            target_idx_val = (targets_val == 0).nonzero()
        else:
            target_idx_train = torch.cat((target_idx_train, (targets_train == idx).nonzero()), dim=0)
            target_idx_val = torch.cat((target_idx_val, (targets_val == idx).nonzero()), dim=0)

    dataset_train_base = torch.utils.data.Subset(dataset_train, target_idx_train)
    dataset_val_base = torch.utils.data.Subset(dataset_val, target_idx_val)

    train_loader_base = torch.utils.data.DataLoader(dataset=dataset_train_base,
                                                   batch_size=args.batch_size,
                                                   shuffle=True, **kwargs)
    val_loader_base = torch.utils.data.DataLoader(dataset_val_base,
                                                 batch_size=args.batch_size,
                                                 shuffle=True, **kwargs)

    # datasets for classes 50-99
    IL_dataset_train = []
    IL_dataset_val = []
    if args.incremental:
        if args.phase >0:
            nc_each = (n_classes - args.baseclass) // args.phase
        else:
            nc_each = 0
        for phase in range(args.phase):
            for idx in range(args.baseclass + phase * nc_each, args.baseclass + (phase + 1) * nc_each):
                if idx == args.baseclass + phase * nc_each:
                    target_idx_train = (targets_train == args.baseclass + phase * nc_each).nonzero()
                    target_idx_val = (targets_val == args.baseclass + phase * nc_each).nonzero()
                else:
                    target_idx_train = torch.cat((target_idx_train, (targets_train == idx).nonzero()), dim=0)
                    target_idx_val = torch.cat((target_idx_val, (targets_val == idx).nonzero()), dim=0)
            dataset_train_f = torch.utils.data.Subset(dataset_train, target_idx_train)
            dataset_val_f = torch.utils.data.Subset(dataset_val, target_idx_val)

            if args.subset:
                train_loader_f = small_batch_dataloader(dataset=dataset_train_f, num_classes=nc_each,
                                                        num_samples=args.shot, batch_size=args.batch_size,
                                                        suffle=True)
            else:
                train_loader_f = torch.utils.data.DataLoader(dataset=dataset_train_f,
                                                             batch_size=args.batch_size,
                                                             shuffle=True, **kwargs)
            val_loader_f = torch.utils.data.DataLoader(dataset_val_f,
                                                       batch_size=args.batch_size,
                                                       shuffle=True, **kwargs)
            IL_dataset_train.append(train_loader_f)
            IL_dataset_val.append(val_loader_f)

    return train_loader_base, val_loader_base, IL_dataset_train, IL_dataset_val
