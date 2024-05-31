import torch

# dataloader

def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms
    
    dataset = datasets.MNIST(
        root='./data', 
        train=is_train, 
        download=True, 
        transform=transforms.Compose([transforms.ToTensor()]))
    
    x = dataset.data.float() / 255.
    y = dataset.targets
    
    if flatten:
        x = x.view(x.size(0), -1)
    
    return x, y

# train_val_split
def split_data(x,y,train_ratio=0.8):
    train_cnt = int(x.size(0)*train_ratio)
    valid_cnt = int(x.size(0)) - train_cnt
    
    # shuffle dataset
    indices = torch.randperm(x.size(0))
    
    x = torch.index_select(x, dim=0, index=indices).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(y, dim=0, index=indices).split([train_cnt, valid_cnt], dim=0)
    
    return x, y
    