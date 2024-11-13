import torch
import torchvision
import numpy as np
import torchvision.transforms as tt

from sklearn.model_selection import train_test_split

from logger.main_logger import MainLogger


class OneHotEncoder:
    def __init__(self, label_count):
        self.label_count = label_count
    
    def __call__(self, label):
        return torch.FloatTensor([1 if i == label else 0 for i in range(self.label_count)])


class CGANDataPreprocessor:
    def __init__(self, args):
        self.__logger = MainLogger(args)
        
        self.batch_size = args.batch_size
        self.num_worker = args.num_worker
        
        self.data_mean, self.data_std = self.__data_mean_std()
        
        self.__trainset = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=None)
        self.__inceptionset = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=None)
        
        self.__logger.debug('data preprocessor init')
        
    def __data_mean_std(self):
        data = torchvision.datasets.CIFAR100("./data", train=True, download=True)
        x = np.concatenate([np.asarray(data[i][0]) for i in range(len(data))])

        mean = np.mean(x, axis=(0, 1))/255
        std = np.std(x, axis=(0, 1))/255

        mean = mean.tolist()
        std = std.tolist()
        self.__logger.debug(f'data mean: {mean}\tdata std: {std}')
        return mean, std
    
    def transform_data(self):
        self.__trainset.transform = tt.Compose([
            tt.Resize(64),
            # tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
            # tt.RandomHorizontalFlip(), 
            tt.ToTensor(),
            tt.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        self.__inceptionset.transform = tt.Compose([
            tt.Resize((299, 299)),
            tt.ToTensor(),
            tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        label_count = len(self.__trainset.classes)
        self.__trainset.target_transform = OneHotEncoder(label_count=label_count)
        self.__inceptionset.target_transform = OneHotEncoder(label_count=label_count)
            
        self.__logger.debug(f'data transform')
        
    # def split_data(self, p=0.2):
    #     targets = self.__trainset.targets

    #     train_idx, val_idx = [], []
    #     for cls in range(100):
    #         cls_idx = [i for i, t in enumerate(targets) if t == cls]
    #         cls_train_idx, cls_val_idx = train_test_split(cls_idx, test_size=p, random_state=42)
    #         train_idx.extend(cls_train_idx)
    #         val_idx.extend(cls_val_idx)

    #     train_dataset_original = torch.utils.data.Subset(self.__trainset, list(range(len(targets))))
    #     val_dataset_original = torch.utils.data.Subset(self.__trainset, val_idx)

    #     self.__train_dataset = torch.utils.data.ConcatDataset([train_dataset_original])
    #     self.__val_dataset = torch.utils.data.ConcatDataset([val_dataset_original])

    #     self.__logger.debug(f'data split - train size: {len(self.__train_dataset)}\tvalidation size: {len(self.__val_dataset)}')

    def get_data_loader(self):
        trainloader = torch.utils.data.DataLoader(self.__trainset, self.batch_size, shuffle=True, num_workers=self.num_worker, pin_memory=True)
        inceptionloader = torch.utils.data.DataLoader(self.__inceptionset, self.batch_size * 2, pin_memory=True, num_workers=0)
        
        self.trainloader = trainloader
        self.inceptionloader = inceptionloader
        return self.trainloader, self.inceptionloader
