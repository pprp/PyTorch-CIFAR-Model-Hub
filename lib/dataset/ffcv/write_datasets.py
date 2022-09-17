import torchvision.datasets as datasets
import torchvision.transforms as transforms
from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter

# Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
train_dataset = datasets.CIFAR10(
    root='~/data',
    train=True,
    download=True,
)
test_dataset = datasets.CIFAR10(
    root='~/data',
    train=False,
    download=True,
)
# write_path = "/home/pdluser/data/cifar10/train_ds.beton"
write_path = '/home/pdluser/data/cifar10/test_ds.beton'

# Pass a type for each data field
writer = DatasetWriter(
    write_path,
    {
        # Tune options to optimize dataset size, throughput at train-time
        'image': RGBImageField(),
        'label': IntField(),
    },
)

# Write dataset
# writer.from_indexed_dataset(train_dataset)
writer.from_indexed_dataset(test_dataset)
