import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from cyclegan import *
from utils import *

Tensor = torch.Tensor


def generate_image(target_type, img_path, output_path):

    class Hyperparameters(object):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    hp = Hyperparameters(
        epoch=0,
        n_epochs=200,
        dataset_train_mode="train",
        dataset_test_mode="test",
        batch_size=4,
        lr=0.0002,
        decay_start_epoch=100,
        b1=0.5,
        b2=0.999,
        n_cpu=8,
        img_size=128,
        channels=3,
        n_critic=5,
        sample_interval=100,
        num_residual_blocks=19,
        lambda_cyc=10.0,
        lambda_id=5.0,
    )

    input_shape = (hp.channels, hp.img_size, hp.img_size)

    Gen_AB = GeneratorResNet(input_shape, hp.num_residual_blocks)
    Gen_BA = GeneratorResNet(input_shape, hp.num_residual_blocks)

    load_checkpoint("./latest.pth", Gen_BA, Gen_AB)

    transforms_ = [
        transforms.Resize((hp.img_size, hp.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    val_dataloader = DataLoader(
        ImageDataset(
            img_path,
            mode=hp.dataset_test_mode,
            transforms_=transforms_,
            unaligned=False,
        ),
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )

    imgs = next(iter(val_dataloader))

    Gen_AB.eval()
    Gen_BA.eval()

    if target_type == "B":
        real_A = Variable(imgs["img"].type(Tensor))
        fake_B = Gen_AB(real_A)
        real_A = make_grid(real_A, nrow=1, normalize=True)
        fake_B = make_grid(fake_B, nrow=1, normalize=True)
        gen_img = torch.cat([fake_B], 1)
        save_image(gen_img, output_path, normalize=False)

    else:
        real_B = Variable(imgs["img"].type(Tensor))
        fake_A = Gen_BA(real_B)
        real_B = make_grid(real_B, nrow=1, normalize=True)
        fake_A = make_grid(fake_A, nrow=1, normalize=True)

        gen_img = torch.cat([fake_A], 1)
        save_image(gen_img, output_path, normalize=False)


def load_checkpoint(
    checkpoint_path,
    Gen_BA,
    Gen_AB,
):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    Gen_BA.load_state_dict(checkpoint["Gen_BA_state_dict"])
    Gen_AB.load_state_dict(checkpoint["Gen_AB_state_dict"])
    print("Model loaded")


# testA - CT, testB - MRI
# B - MRI, A - CT


def generate_img_wrapper(target_type, input_file, output_file):
    if target_type == "mri":
        generate_image(target_type="B", img_path=input_file, output_path=output_file)
    else:
        generate_image(target_type="A", img_path=input_file, output_path=output_file)


if __name__ == "__main__":
    generate_image(
        target_type="A",
        img_path="../split-images/testB/0001.png",
        output_path="./output.png",
    )
