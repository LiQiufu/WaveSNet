from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        class_names = train_set.class_names
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class, class_names

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        class_names = train_set.class_names
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class, class_names

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        class_names = train_set.class_names
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class, class_names

    else:
        raise NotImplementedError

