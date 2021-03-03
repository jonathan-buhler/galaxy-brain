from torch.utils.data import DataLoader
from datasets import HDG10


dataset = HDG10("./src/datasets/HDG10.h5")
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for i_batch, (imgs, labels) in enumerate(data_loader):
    print(imgs[0])
    # print(labels)
    break
