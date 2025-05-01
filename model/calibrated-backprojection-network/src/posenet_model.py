'''
Author: Alex Wong <alexw@cs.ucla.edu>

If you use this code, please cite the following paper:

A. Wong, and S. Soatto. Unsupervised Depth Completion with Calibrated Backprojection Layers.
https://arxiv.org/pdf/2108.10531.pdf

@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
'''
import torch, torchvision
import losses, networks, net_utils, log_utils # Added from kbnet_model.py: losses, net_utils

EPSILON = 1e-8


class PoseNetModel(object):
    '''
    Pose network for computing relative pose between a pair of images

    Arg(s):
        encoder_type : str
            posenet, resnet18, resnet34
        rotation_parameterization : str
            rotation parameterization: axis
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function for network
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 encoder_type='posenet',
                 rotation_parameterization='axis',
                 weight_initializer='xavier_normal',
                 activation_func='leaky_relu',
                 device=torch.device('cuda')):

        self.device = device

        # Create pose encoder
        if encoder_type == 'posenet':
            self.encoder = networks.PoseEncoder(
                input_channels=6,
                n_filters=[16, 32, 64, 128, 256, 256, 256],
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=True)
        elif encoder_type == 'resnet18':
            self.encoder = networks.ResNetEncoder(
                n_layer=18,
                input_channels=6,
                n_filters=[16, 32, 64, 128, 256],
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=True)
        elif encoder_type == 'resnet34':
            self.encoder = networks.ResNetEncoder(
                n_layer=34,
                input_channels=6,
                n_filters=[16, 32, 64, 128, 256],
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=True)
        else:
            raise ValueError('Unsupported PoseNet encoder type: {}'.format(encoder_type))

        # Create pose decoder
        if encoder_type == 'posenet':
            self.decoder = networks.PoseDecoder(
                rotation_parameterization=rotation_parameterization,
                weight_initializer=weight_initializer,
                input_channels=256)
        elif encoder_type == 'resnet18' or encoder_type == 'resnet34':
            self.decoder = networks.PoseDecoder(
                rotation_parameterization=rotation_parameterization,
                input_channels=256,
                n_filters=[256, 256],
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=True)
        else:
            raise ValueError('Unsupported PoseNet encoder type: {}'.format(encoder_type))

        # Move to device
        self.data_parallel()
        self.to(self.device)

    def forward(self, image0, image1):
        '''
        Forwards the inputs through the network

        Arg(s):
            image0 : torch.Tensor[float32]
                image at time step 0
            image1 : torch.Tensor[float32]
                image at time step 1
        Returns:
            torch.Tensor[float32] : pose from time step 1 to 0
        '''

        # Forward through the network
        latent, _ = self.encoder(torch.cat([image0, image1], dim=1))
        output = self.decoder(latent)

        return output

    def parameters(self):
        '''
        Returns the set of parameters
        '''

        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def train(self):
        '''
        Sets model to training mode
        '''

        self.encoder.train()
        self.decoder.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.encoder.eval()
        self.decoder.eval()

    def to(self, device):
        '''
        Moves model to specified device

        Arg(s):
            device : torch.device
                device for running model
        '''

        # Move to device
        self.encoder.to(device)
        self.decoder.to(device)

    def save_model(self, checkpoint_path, step, optimizer):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        checkpoint = {}
        # Save training state
        checkpoint['train_step'] = step
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Save encoder and decoder weights
        checkpoint['encoder_state_dict'] = self.encoder.state_dict()
        checkpoint['decoder_state_dict'] = self.decoder.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def restore_model(self, checkpoint_path, optimizer=None):
        '''
        Restore weights of the model

        Arg(s):
            checkpoint_path : str
                path to checkpoint
            optimizer : torch.optim
                optimizer
        Returns:
            int : current step in optimization
            torch.optim : optimizer with restored state
        '''

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore encoder and decoder weights
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Return the current step and optimizer
        return checkpoint['train_step'], optimizer

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder = torch.nn.DataParallel(self.decoder)


    '''
    ============================================= Below modified from kbnet_model.py =============================================
    '''

    def compute_loss(self,
                     image0,
                     image1,
                     depth0, #input depth
                    #  inpaint_depth0,
                     validity_map_depth0,
                     intrinsics0,
                     intrinsics1,
                     pose01,
                    #  pose02,
                     w_color=0.15,
                     w_structure=0.95):
                    #  w_sparse_depth=0.60,
                    #  w_smoothness=0.04):
        '''
        Computes loss function
        l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}

        Arg(s):
            image0 : torch.Tensor[float32]
                N x 3 x H x W image at time step t
            image1 : torch.Tensor[float32]
                N x 3 x H x W image at time step t-1
            depth0 : torch.Tensor[float32]
                N x 1 x H x W depth at time t
            validity_map_depth0 : torch.Tensor[float32]
                N x 1 x H x W depth at time t
            intrinsics0 : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix at time t
            intrinsics1 : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix at time t-1
            pose01 : torch.Tensor[float32]
                N x 4 x 4 relative pose from image at time t to t-1
            w_color : float
                weight of color consistency term
            w_structure : float
                weight of structure consistency term (SSIM)
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        shape = image0.shape
        validity_map_image0 = torch.ones_like(depth0)
        #validity_map_image0 = torch.ones_like(validity_map_depth0)

    # Backproject points to 3D camera coordinates
        points = net_utils.backproject_to_camera(depth0, intrinsics0, shape)

        # Reproject points onto image 1 and image 2
        target_xy01 = net_utils.project_to_pixel(points, pose01, intrinsics1, shape)

        # Reconstruct image0 from image1 and image2 by reprojection
        image01 = net_utils.grid_sample(image1, target_xy01, shape)

        '''
        Essential loss terms
        '''
        # Color consistency loss function
        loss_color01 = losses.color_consistency_loss_func(
            src=image01,
            tgt=image0,
            w=validity_map_image0)
        # loss_color02 = losses.color_consistency_loss_func(
        #     src=image02,
        #     tgt=image0,
        #     w=validity_map_image0)
        # loss_color = loss_color01 + loss_color02
        loss_color = loss_color01

        # don't need smoothness term; just need color and structure

        # Structural consistency loss function
        loss_structure01 = losses.structural_consistency_loss_func(
            src=image01,
            tgt=image0,
            w=validity_map_image0)
        # loss_structure02 = losses.structural_consistency_loss_func(
        #     src=image02,
        #     tgt=image0,
        #     w=validity_map_image0)
        loss_structure = loss_structure01 # + loss_structure02

        # Sparse depth consistency loss function
        # loss_sparse_depth = losses.sparse_depth_consistency_loss_func(
        #     src=output_depth0,
        #     tgt=sparse_depth0,
        #     w=validity_map_depth0)

        # Local smoothness loss function
        # loss_smoothness = losses.smoothness_loss_func(
        #     predict=output_depth0,
        #     image=image0)

        # l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}
        loss = w_color * loss_color + \
            w_structure * loss_structure # + \
            # w_sparse_depth * loss_sparse_depth # + \
            # w_smoothness * loss_smoothness

        loss_info = {
            'loss_color' : loss_color,
            'loss_structure' : loss_structure,
            # 'loss_sparse_depth' : loss_sparse_depth,
            # 'loss_smoothness' : loss_smoothness,
            'loss' : loss,
            'image01' : image01
            # 'image02' : image02
        }

        return loss, loss_info

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image0=None,
                    image01=None,
                    # image02=None,
                    # output_depth0=None,
                    # sparse_depth0=None,
                    # validity_map0=None,
                    # ground_truth0=None,
                    pose01=None,
                    #pose02=None,
                    scalars={},
                    n_display=4):
        '''
        Logs summary to Tensorboard

        Arg(s):
            summary_writer : SummaryWriter
                Tensorboard summary writer
            tag : str
                tag that prefixes names to log
            step : int
                current step in training
            image0 : torch.Tensor[float32]
                image at time step t
            image01 : torch.Tensor[float32]
                image at time step t-1 warped to time step t
            image02 : torch.Tensor[float32]
                image at time step t+1 warped to time step t
            output_depth0 : torch.Tensor[float32]
                output depth at time t
            sparse_depth0 : torch.Tensor[float32]
                sparse_depth at time t
            validity_map0 : torch.Tensor[float32]
                validity map of sparse depth at time t
            ground_truth0 : torch.Tensor[float32]
                ground truth depth at time t
            pose01 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t to t-1
            pose02 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t to t+1
            scalars : dict[str, float]
                dictionary of scalars to log
            n_display : int
                number of images to display
        '''

        with torch.no_grad():

            display_summary_image = []
            display_summary_depth = []

            display_summary_image_text = tag
            display_summary_depth_text = tag

            if image0 is not None:
                image0_summary = image0[0:n_display, ...]

                display_summary_image_text += '_image0'
                display_summary_depth_text += '_image0'

                # Add to list of images to log
                display_summary_image.append(
                    torch.cat([
                        image0_summary.cpu(),
                        torch.zeros_like(image0_summary, device=torch.device('cpu'))],
                        dim=-1))

                display_summary_depth.append(display_summary_image[-1])

            if image0 is not None and image01 is not None:
                image01_summary = image01[0:n_display, ...]

                display_summary_image_text += '_image01-error'

                # Compute reconstruction error w.r.t. image 0
                image01_error_summary = torch.mean(
                    torch.abs(image0_summary - image01_summary),
                    dim=1,
                    keepdim=True)

                # Add to list of images to log
                image01_error_summary = log_utils.colorize(
                    (image01_error_summary / 0.10).cpu(),
                    colormap='inferno')

                display_summary_image.append(
                    torch.cat([
                        image01_summary.cpu(),
                        image01_error_summary],
                        dim=3))

            # if image0 is not None and image02 is not None:
            #     image02_summary = image02[0:n_display, ...]
            #
            #     display_summary_image_text += '_image02-error'
            #
            #     # Compute reconstruction error w.r.t. image 0
            #     image02_error_summary = torch.mean(
            #         torch.abs(image0_summary - image02_summary),
            #         dim=1,
            #         keepdim=True)
            #
            #     # Add to list of images to log
            #     image02_error_summary = log_utils.colorize(
            #         (image02_error_summary / 0.10).cpu(),
            #         colormap='inferno')
            #
            #     display_summary_image.append(
            #         torch.cat([
            #             image02_summary.cpu(),
            #             image02_error_summary],
            #             dim=3))
            #
            # if output_depth0 is not None:
            #     output_depth0_summary = output_depth0[0:n_display, ...]
            #
            #     display_summary_depth_text += '_output0'
            #
            #     # Add to list of images to log
            #     n_batch, _, n_height, n_width = output_depth0_summary.shape
            #
            #     display_summary_depth.append(
            #         torch.cat([
            #             log_utils.colorize(
            #                 (output_depth0_summary / self.max_predict_depth).cpu(),
            #                 colormap='viridis'),
            #             torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
            #             dim=3))
            #
            #     # Log distribution of output depth
            #     summary_writer.add_histogram(tag + '_output_depth0_distro', output_depth0, global_step=step)
            #
            # if output_depth0 is not None and sparse_depth0 is not None and validity_map0 is not None:
            #     sparse_depth0_summary = sparse_depth0[0:n_display]
            #     validity_map0_summary = validity_map0[0:n_display]
            #
            #     display_summary_depth_text += '_sparse0-error'
            #
            #     # Compute output error w.r.t. input sparse depth
            #     sparse_depth0_error_summary = \
            #         torch.abs(output_depth0_summary - sparse_depth0_summary)
            #
            #     sparse_depth0_error_summary = torch.where(
            #         validity_map0_summary == 1.0,
            #         (sparse_depth0_error_summary + EPSILON) / (sparse_depth0_summary + EPSILON),
            #         validity_map0_summary)
            #
            #     # Add to list of images to log
            #     sparse_depth0_summary = log_utils.colorize(
            #         (sparse_depth0_summary / self.max_predict_depth).cpu(),
            #         colormap='viridis')
            #     sparse_depth0_error_summary = log_utils.colorize(
            #         (sparse_depth0_error_summary / 0.05).cpu(),
            #         colormap='inferno')
            #
            #     display_summary_depth.append(
            #         torch.cat([
            #             sparse_depth0_summary,
            #             sparse_depth0_error_summary],
            #             dim=3))
            #
            #     # Log distribution of sparse depth
            #     summary_writer.add_histogram(tag + '_sparse_depth0_distro', sparse_depth0, global_step=step)
            #
            # if output_depth0 is not None and ground_truth0 is not None:
            #     validity_map0 = torch.unsqueeze(ground_truth0[:, 1, :, :], dim=1)
            #     ground_truth0 = torch.unsqueeze(ground_truth0[:, 0, :, :], dim=1)
            #
            #     validity_map0_summary = validity_map0[0:n_display]
            #     ground_truth0_summary = ground_truth0[0:n_display]
            #
            #     display_summary_depth_text += '_groundtruth0-error'
            #
            #     # Compute output error w.r.t. ground truth
            #     ground_truth0_error_summary = \
            #         torch.abs(output_depth0_summary - ground_truth0_summary)
            #
            #     ground_truth0_error_summary = torch.where(
            #         validity_map0_summary == 1.0,
            #         (ground_truth0_error_summary + EPSILON) / (ground_truth0_summary + EPSILON),
            #         validity_map0_summary)
            #
            #     # Add to list of images to log
            #     ground_truth0_summary = log_utils.colorize(
            #         (ground_truth0_summary / self.max_predict_depth).cpu(),
            #         colormap='viridis')
            #     ground_truth0_error_summary = log_utils.colorize(
            #         (ground_truth0_error_summary / 0.05).cpu(),
            #         colormap='inferno')
            #
            #     display_summary_depth.append(
            #         torch.cat([
            #             ground_truth0_summary,
            #             ground_truth0_error_summary],
            #             dim=3))
            #
            #     # Log distribution of ground truth
            #     summary_writer.add_histogram(tag + '_ground_truth0_distro', ground_truth0, global_step=step)

            if pose01 is not None:
                # Log distribution of pose 1 to 0translation vector
                summary_writer.add_histogram(tag + '_tx01_distro', pose01[:, 0, 3], global_step=step)
                summary_writer.add_histogram(tag + '_ty01_distro', pose01[:, 1, 3], global_step=step)
                summary_writer.add_histogram(tag + '_tz01_distro', pose01[:, 2, 3], global_step=step)

            # if pose02 is not None:
            #     # Log distribution of pose 2 to 0 translation vector
            #     summary_writer.add_histogram(tag + '_tx02_distro', pose02[:, 0, 3], global_step=step)
            #     summary_writer.add_histogram(tag + '_ty02_distro', pose02[:, 1, 3], global_step=step)
            #     summary_writer.add_histogram(tag + '_tz02_distro', pose02[:, 2, 3], global_step=step)

        # Log scalars to tensorboard
        for (name, value) in scalars.items():
            summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

        # Log image summaries to tensorboard
        if len(display_summary_image) > 1:
            display_summary_image = torch.cat(display_summary_image, dim=2)

            summary_writer.add_image(
                display_summary_image_text,
                torchvision.utils.make_grid(display_summary_image, nrow=n_display),
                global_step=step)

        if len(display_summary_depth) > 1:
            display_summary_depth = torch.cat(display_summary_depth, dim=2)

            summary_writer.add_image(
                display_summary_depth_text,
                torchvision.utils.make_grid(display_summary_depth, nrow=n_display),
                global_step=step)
