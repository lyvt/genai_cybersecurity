import torch


class Generator(torch.nn.Module):
    def __init__(self, input_dim, label_dim, num_filters, output_dim):
        super(Generator, self).__init__()

        # Hidden layers for input and label
        self.hidden_layer1 = torch.nn.Sequential()
        self.hidden_layer2 = torch.nn.Sequential()
        self.hidden_layers = torch.nn.Sequential()

        for i in range(len(num_filters)):
            if i == 0:
                # Input layer
                input_deconv = torch.nn.ConvTranspose2d(input_dim, num_filters[i] // 2, kernel_size=4, stride=1,
                                                        padding=0)
                self.hidden_layer1.add_module('input_deconv', input_deconv)
                torch.nn.init.normal_(input_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(input_deconv.bias, 0.0)
                self.hidden_layer1.add_module('input_bn', torch.nn.BatchNorm2d(num_filters[i] // 2))
                self.hidden_layer1.add_module('input_act', torch.nn.ReLU())

                # Label layer
                label_deconv = torch.nn.ConvTranspose2d(label_dim, num_filters[i] // 2, kernel_size=4, stride=1,
                                                        padding=0)
                self.hidden_layer2.add_module('label_deconv', label_deconv)
                torch.nn.init.normal_(label_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(label_deconv.bias, 0.0)
                self.hidden_layer2.add_module('label_bn', torch.nn.BatchNorm2d(num_filters[i] // 2))
                self.hidden_layer2.add_module('label_act', torch.nn.ReLU())
            else:
                # Deconvolution layers
                deconv = torch.nn.ConvTranspose2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2,
                                                  padding=1)
                self.hidden_layers.add_module(f'deconv_{i}', deconv)
                torch.nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(deconv.bias, 0.0)
                self.hidden_layers.add_module(f'bn_{i}', torch.nn.BatchNorm2d(num_filters[i]))
                self.hidden_layers.add_module(f'act_{i}', torch.nn.ReLU())

        # Output layer
        self.output_layer = torch.nn.Sequential()
        out_deconv = torch.nn.ConvTranspose2d(num_filters[-1], output_dim, kernel_size=4, stride=2, padding=1)
        self.output_layer.add_module('output_deconv', out_deconv)
        torch.nn.init.normal_(out_deconv.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out_deconv.bias, 0.0)
        self.output_layer.add_module('output_act', torch.nn.Tanh())

    def forward(self, z, c):
        h1 = self.hidden_layer1(z)
        h2 = self.hidden_layer2(c)
        x = torch.cat([h1, h2], dim=1)
        h = self.hidden_layers(x)
        out = self.output_layer(h)
        return out


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, label_dim, num_filters, output_dim):
        super(Discriminator, self).__init__()

        # Hidden layers for input and label
        self.hidden_layer1 = torch.nn.Sequential()
        self.hidden_layer2 = torch.nn.Sequential()
        self.hidden_layers = torch.nn.Sequential()

        for i in range(len(num_filters)):
            if i == 0:
                # Input layer
                input_conv = torch.nn.Conv2d(input_dim, num_filters[i] // 2, kernel_size=4, stride=2, padding=1)
                self.hidden_layer1.add_module('input_conv', input_conv)
                torch.nn.init.normal_(input_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(input_conv.bias, 0.0)
                self.hidden_layer1.add_module('input_act', torch.nn.LeakyReLU(0.2))

                # Label layer
                label_conv = torch.nn.Conv2d(label_dim, num_filters[i] // 2, kernel_size=4, stride=2, padding=1)
                self.hidden_layer2.add_module('label_conv', label_conv)
                torch.nn.init.normal_(label_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(label_conv.bias, 0.0)
                self.hidden_layer2.add_module('label_act', torch.nn.LeakyReLU(0.2))
            else:
                # Convolutional layers
                conv = torch.nn.Conv2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1)
                self.hidden_layers.add_module(f'conv_{i}', conv)
                torch.nn.init.normal_(conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(conv.bias, 0.0)
                self.hidden_layers.add_module(f'bn_{i}', torch.nn.BatchNorm2d(num_filters[i]))
                self.hidden_layers.add_module(f'act_{i}', torch.nn.LeakyReLU(0.2))

        # Output layer
        self.output_layer = torch.nn.Sequential()
        out_conv = torch.nn.Conv2d(num_filters[-1], output_dim, kernel_size=4, stride=1, padding=0)
        self.output_layer.add_module('output_conv', out_conv)
        torch.nn.init.normal_(out_conv.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out_conv.bias, 0.0)
        self.output_layer.add_module('output_act', torch.nn.Sigmoid())

    def forward(self, z, c):
        h1 = self.hidden_layer1(z)
        h2 = self.hidden_layer2(c)
        x = torch.cat([h1, h2], dim=1)
        h = self.hidden_layers(x)
        out = self.output_layer(h)
        return out
