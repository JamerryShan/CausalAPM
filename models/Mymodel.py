import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertModel


class MyAutoModel(nn.Module):
    def __init__(self, pretrained_path, config, args=None, cls_num=3):
        super(MyAutoModel, self).__init__()

        self.cls_num = cls_num
        self.bert_config = config
        self.hidden_size = self.bert_config.hidden_size
        self.ae_z_size = args.ae_z_size
        self.is_detach = args.is_detach
        self.z_split_first_size = args.z_split_first_size

        self.bert = BertModel.from_pretrained(pretrained_path)
        print(self.bert.config)

        self.ae_encoder = nn.Linear(self.hidden_size, self.ae_z_size)

        self.fc_fx = nn.Linear(self.ae_z_size, cls_num)
        self.fc_z1 = nn.Linear(self.z_split_first_size, 1)
        self.fc_z2 = nn.Linear(self.ae_z_size - self.z_split_first_size, self.z_split_first_size)

        # self.fc_x_minus_fx2mnli = nn.Linear(self.hidden_size, cls_num)

        # 4/16改动 add two fc: z1->pred, z2->pred
        self.fc_pd4mz1 = nn.Linear(self.z_split_first_size, self.cls_num)
        self.fc_pd4mz2 = nn.Linear(self.ae_z_size - self.z_split_first_size, self.cls_num)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs[1]
        bert_output_feat = sequence_output # (batch_size, 768)
        bert_feat_detached = bert_output_feat.detach()
        z_bert_detached = self.ae_encoder(bert_feat_detached)

        z = self.ae_encoder(bert_output_feat)

        output_fx = self.fc_fx(z)


        # MyResults/mnli_loss123_12bert.log
        z1, nodetach_z2 = torch.split(z, [self.z_split_first_size, self.ae_z_size - self.z_split_first_size], dim=1)
        z1_nobert, z2 = torch.split(z_bert_detached, [self.z_split_first_size, self.ae_z_size - self.z_split_first_size], dim=1)

        #后四维预测一下看看
        # nodetach_z2, z1 = torch.split(z, [self.ae_z_size - self.z_split_first_size, self.z_split_first_size], dim=1)
        # z2, z1_nobert = torch.split(z_bert_detached, [self.ae_z_size - self.z_split_first_size, self.z_split_first_size], dim=1)

        #
        output_z1_fc = self.fc_z1(z1)
        output_z2_fc = self.fc_z2(z2)
        output_z2_detached = self.fc_z2(z2.detach())
        # 4/14
        
        pd4mz1 = self.fc_pd4mz1(z1.detach())
        pd4mz2 = self.fc_pd4mz2(nodetach_z2.detach())

        return output_fx, output_z1_fc, output_z2_fc, z1_nobert, output_z2_detached, nodetach_z2, pd4mz1, pd4mz2



#
#
# import torch
# from models import BaseVAE
# from torch import nn
# from torch.nn import functional as F
# from typing import List
# from abc import abstractmethod
#
# class BaseVAE(nn.Module):
#
#     def __init__(self):
#         super(BaseVAE, self).__init__()
#
#     def encode(self, input):
#         raise NotImplementedError
#
#     def decode(self, input):
#         raise NotImplementedError
#
#     def sample(self, batch_size, current_device, **kwargs):
#         raise NotImplementedError
#
#     def generate(self, x, **kwargs):
#         raise NotImplementedError
#
#     @abstractmethod
#     def forward(self, *inputs):
#         pass
#
#     @abstractmethod
#     def loss_function(self, *inputs, **kwargs):
#         pass
#
#
#
#
#
# class VanillaVAE(BaseVAE):
#
#     def __init__(self,
#                  in_channels: int,
#                  latent_dim: int,
#                  hidden_dims: List = None,
#                  **kwargs):
#         super(VanillaVAE, self).__init__()
#
#         self.latent_dim = latent_dim
#
#         modules = []
#         if hidden_dims is None:
#             hidden_dims = [32, 64, 128, 256, 512]
#
#         # Build Encoder
#         for h_dim in hidden_dims:
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels=h_dim,
#                               kernel_size= 3, stride= 2, padding  = 1),
#                     nn.BatchNorm2d(h_dim),
#                     nn.LeakyReLU())
#             )
#             in_channels = h_dim
#
#         self.encoder = nn.Sequential(*modules)
#         self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
#         self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
#
#
#         # Build Decoder
#         modules = []
#
#         self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
#
#         hidden_dims.reverse()
#
#         for i in range(len(hidden_dims) - 1):
#             modules.append(
#                 nn.Sequential(
#                     nn.ConvTranspose2d(hidden_dims[i],
#                                        hidden_dims[i + 1],
#                                        kernel_size=3,
#                                        stride = 2,
#                                        padding=1,
#                                        output_padding=1),
#                     nn.BatchNorm2d(hidden_dims[i + 1]),
#                     nn.LeakyReLU())
#             )
#
#
#
#         self.decoder = nn.Sequential(*modules)
#
#         self.final_layer = nn.Sequential(
#                             nn.ConvTranspose2d(hidden_dims[-1],
#                                                hidden_dims[-1],
#                                                kernel_size=3,
#                                                stride=2,
#                                                padding=1,
#                                                output_padding=1),
#                             nn.BatchNorm2d(hidden_dims[-1]),
#                             nn.LeakyReLU(),
#                             nn.Conv2d(hidden_dims[-1], out_channels= 3,
#                                       kernel_size= 3, padding= 1),
#                             nn.Tanh())
#
#     def encode(self, input):
#         """
#         Encodes the input by passing through the encoder network
#         and returns the latent codes.
#         :param input: (Tensor) Input tensor to encoder [N x C x H x W]
#         :return: (Tensor) List of latent codes
#         """
#         result = self.encoder(input)
#         result = torch.flatten(result, start_dim=1)
#
#         # Split the result into mu and var components
#         # of the latent Gaussian distribution
#         mu = self.fc_mu(result)
#         log_var = self.fc_var(result)
#
#         return [mu, log_var]
#
#     def decode(self, z):
#         """
#         Maps the given latent codes
#         onto the image space.
#         :param z: (Tensor) [B x D]
#         :return: (Tensor) [B x C x H x W]
#         """
#         result = self.decoder_input(z)
#         result = result.view(-1, 512, 2, 2)
#         result = self.decoder(result)
#         result = self.final_layer(result)
#         return result
#
#     def reparameterize(self, mu, logvar):
#         """
#         Reparameterization trick to sample from N(mu, var) from
#         N(0,1).
#         :param mu: (Tensor) Mean of the latent Gaussian [B x D]
#         :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
#         :return: (Tensor) [B x D]
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu
#
#     def forward(self, input, **kwargs):
#         mu, log_var = self.encode(input)
#         z = self.reparameterize(mu, log_var)
#         return  [self.decode(z), input, mu, log_var]
#
#     def loss_function(self,
#                       *args,
#                       **kwargs):
#         """
#         Computes the VAE loss function.
#         KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
#         :param args:
#         :param kwargs:
#         :return:
#         """
#         recons = args[0]
#         input = args[1]
#         mu = args[2]
#         log_var = args[3]
#
#         kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
#         recons_loss =F.mse_loss(recons, input)
#
#
#         kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
#
#         loss = recons_loss + kld_weight * kld_loss
#         return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
#
#     def sample(self,
#                num_samples:int,
#                current_device: int, **kwargs):
#         """
#         Samples from the latent space and return the corresponding
#         image space map.
#         :param num_samples: (Int) Number of samples
#         :param current_device: (Int) Device to run the model
#         :return: (Tensor)
#         """
#         z = torch.randn(num_samples,
#                         self.latent_dim)
#
#         z = z.to(current_device)
#
#         samples = self.decode(z)
#         return samples
#
#     def generate(self, x, **kwargs):
#         """
#         Given an input image x, returns the reconstructed image
#         :param x: (Tensor) [B x C x H x W]
#         :return: (Tensor) [B x C x H x W]
#         """
#
#         return self.forward(x)[0]