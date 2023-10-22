import torch
import torch.nn as nn
import kornia

class LSTMFusion(torch.nn.Module):
    def __init__(self, base_channels=8):
        super(LSTMFusion, self).__init__()
        # C = 24
        # self.lstm_cell = MVSLayernormConvLSTMCell(input_dim=base_channels * 5,
        #                                           hidden_dim=base_channels * 5,
        #                                           kernel_size=(3, 3),
        #                                           activation_function=torch.celu)

        self.lstm_cell = MVSLayernormConvLSTMCell(input_dim=base_channels * 3,
                                                  hidden_dim=base_channels * 3,
                                                  kernel_size=(3, 3),
                                                  activation_function=torch.celu)

    def forward(self, current_encoding, current_state, previous_pose, current_pose, estimated_current_depth,
                camera_matrix):
        batch, channel, height, width = current_encoding.size()

        if current_state is None:
            hidden_state, cell_state = self.lstm_cell.init_hidden(batch_size=batch,
                                                                  image_size=(height, width))
        else:
            hidden_state, cell_state = current_state

        next_hidden_state, next_cell_state = self.lstm_cell(input_tensor=current_encoding,
                                                            cur_state=[hidden_state, cell_state],
                                                            previous_pose=previous_pose,
                                                            current_pose=current_pose,
                                                            estimated_current_depth=estimated_current_depth,
                                                            camera_matrix=camera_matrix)

        return next_hidden_state, next_cell_state


class MVSLayernormConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, activation_function=None):
        super(MVSLayernormConvLSTMCell, self).__init__()

        self.activation_function = activation_function

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=False)

    def forward(self, input_tensor, cur_state, previous_pose, current_pose, estimated_current_depth, camera_matrix):
        h_cur, c_cur = cur_state

        # if previous_pose is not None:
        #     transformation = torch.bmm(torch.inverse(previous_pose), current_pose)
        #
        #     non_valid = estimated_current_depth <= 0.01
        #     h_cur, mask = warp_frame_depth(image_src=h_cur,
        #                              depth_dst=estimated_current_depth,
        #                              src_trans_dst=transformation,
        #                              camera_matrix=camera_matrix,
        #                              normalize_points=False,
        #                              sampling_mode='bilinear')
        #     b, c, h, w = h_cur.size()
        #     non_valid = torch.cat([non_valid] * c, dim=1)
        #     h_cur.data[non_valid] = 0.0

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        b, c, h, w = h_cur.size()
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)

        cc_g = torch.layer_norm(cc_g, [h, w])
        g = self.activation_function(cc_g)

        c_next = f * c_cur + i * g
        c_next = torch.layer_norm(c_next, [h, w])
        h_next = o * self.activation_function(c_next)

        # (B, hidden_size, H, W), (B, hidden_size, H, W)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


def warp_frame_depth(
        image_src: torch.Tensor,
        depth_dst: torch.Tensor,
        src_trans_dst: torch.Tensor,
        camera_matrix: torch.Tensor,
        normalize_points: bool = False,
        sampling_mode='bilinear') -> (torch.Tensor, torch.Tensor):
    # TAKEN FROM KORNIA LIBRARY
    if not isinstance(image_src, torch.Tensor):
        raise TypeError(f"Input image_src type is not a torch.Tensor. Got {type(image_src)}.")

    if not len(image_src.shape) == 4:
        raise ValueError(f"Input image_src musth have a shape (B, D, H, W). Got: {image_src.shape}")

    if not isinstance(depth_dst, torch.Tensor):
        raise TypeError(f"Input depht_dst type is not a torch.Tensor. Got {type(depth_dst)}.")

    if not len(depth_dst.shape) == 4 and depth_dst.shape[-3] == 1:
        raise ValueError(f"Input depth_dst musth have a shape (B, 1, H, W). Got: {depth_dst.shape}")

    if not isinstance(src_trans_dst, torch.Tensor):
        raise TypeError(f"Input src_trans_dst type is not a torch.Tensor. "
                        f"Got {type(src_trans_dst)}.")

    if not len(src_trans_dst.shape) == 3 and src_trans_dst.shape[-2:] == (3, 3):
        raise ValueError(f"Input src_trans_dst must have a shape (B, 3, 3). "
                         f"Got: {src_trans_dst.shape}.")

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(f"Input camera_matrix type is not a torch.Tensor. "
                        f"Got {type(camera_matrix)}.")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). "
                         f"Got: {camera_matrix.shape}.")
    # unproject source points to camera frame
    points_3d_dst: torch.Tensor = kornia.depth_to_3d(depth_dst, camera_matrix, normalize_points)  # Bx3xHxW

    # transform points from source to destination
    points_3d_dst = points_3d_dst.permute(0, 2, 3, 1)  # BxHxWx3

    # apply transformation to the 3d points
    points_3d_src = kornia.transform_points(src_trans_dst[:, None], points_3d_dst)  # BxHxWx3
    mask = (points_3d_src[:, :, :, 2] > 0).unsqueeze(1)

    # project back to pixels
    camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1xHxW
    points_2d_src: torch.Tensor = kornia.project_points(points_3d_src, camera_matrix_tmp)  # BxHxWx2

    # normalize points between [-1 / 1]
    height, width = depth_dst.shape[-2:]
    points_2d_src_norm: torch.Tensor = kornia.normalize_pixel_coordinates(points_2d_src, height, width)  # BxHxWx2

    warp_feature = torch.nn.functional.grid_sample(image_src, points_2d_src_norm, align_corners=True, mode=sampling_mode)

    C = warp_feature.shape[1]
    mask_3 = mask.repeat(1, C, 1, 1)
    warp_feature[~mask_3] = 0.0
    return warp_feature, mask
