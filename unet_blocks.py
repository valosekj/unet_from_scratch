import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define a new class ConvBlock, which is a subclass of nn.Module.
class ConvBlock(nn.Module):
    # This is the constructor of the ConvBlock class. The __init__ method is automatically called when a new instance
    # of ConvBlock is created. It initializes the instance with the provided arguments. Here, in_channels and
    # out_channels are the number of input and output channels for the convolutional layers within the block.
    def __init__(self, in_channels, out_channels):
        # Just inheriting from nn.Module (`ConvBlock(nn.Module)`) doesn't automatically call its __init__ method when
        # creating an instance of ConvBlock. If you want the __init__ method of the superclass (nn.Module) to run, you
        # must call it explicitly. This is where super() in the following command comes in.
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.bridge = ConvBlock(512, 1024)
        self.decoder1 = DecoderBlock(1024, 512, 512)
        self.decoder2 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder4 = DecoderBlock(128, 64, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # s - output of the convolutional block
        # p - output after the max pooling layer
        # d - output of the decoder block
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)
        b = self.bridge(p4)
        d1 = self.decoder1(b, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)
        out = self.final(d4)

        # # Sanity check - images in the batch
        # fig, axes = plt.subplots(1, 2)
        # axes[0].imshow(x[0, :, :, :].squeeze())
        # axes[1].imshow(x[1, :, :, :].squeeze())
        # plt.show()
        #
        # # Sanity check - images through the network
        # fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        # images = [x, p1, p2, p3, p4, b, d1, d2, d3, d4]
        # titles = ['Input Image', 'Encoder1 Output', 'Encoder2 Output', 'Encoder3 Output', 'Encoder4 Output',
        #           'Bridge Output', 'Decoder1 Output', 'Decoder2 Output', 'Decoder3 Output', 'Decoder4 Output']
        # for i, (img, title) in enumerate(zip(images, titles)):
        #     ax = axes[i // 5, i % 5]
        #     ax.imshow(img[0, 0, :, :].detach().numpy().squeeze(), cmap='gray')
        #     ax.set_title(title)
        # plt.tight_layout()
        # plt.show()

        return out
