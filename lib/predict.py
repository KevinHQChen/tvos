import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image

from .utils import idx2onehot

savefigs = False


def predict(ref,
            target,
            ref_label,
            flow_history,
            weight_dense,
            weight_sparse,
            frame_idx,
            args):
    """
    The Predict Function.
    :param ref: (N, feature_dim, H, W)
    :param target: (feature_dim, H, W)
    :param ref_label: (d, N, H*W)
    :param weight_dense: (H*W, H*W)
    :param weight_sparse: (H*W, H*W)
    :param frame_idx:
    :param args:
    :return: (d, H, W)
    """
    # sample frames from history features
    d = ref_label.shape[0]
    sample_idx = sample_frames(frame_idx, args.range, args.ref_num)
    ref_selected = ref.index_select(0, sample_idx)
    ref_label_selected = ref_label.index_select(1, sample_idx).view(d, -1)

    # get similarity matrix
    (num_ref, feature_dim, H, W) = ref_selected.shape
    ref_selected = ref_selected.permute(0, 2, 3, 1).reshape(-1, feature_dim)
    target = target.reshape(feature_dim, -1)
    global_similarity = ref_selected.mm(target)

    # temperature step
    global_similarity *= args.temperature

    if savefigs: print('Appearance prior shape before softmax: {}'.format(global_similarity.shape))

    # softmax
    global_similarity = global_similarity.softmax(dim=0)

    if savefigs: print('Appearance prior shape after softmax: {}'.format(global_similarity.shape))

    (H_, W_, ___) = flow_history[2].shape
    # optical flow
    flow_weight = get_flow_weight(H_, W_, flow_history, sigma=1)

    # spatial weight and motion model
    global_similarity = global_similarity.contiguous().view(num_ref, H * W, H * W)
    if frame_idx > 15:
        continuous_frame = 4
        # sparse spatial prior on older frames (older than 4 frames ago)
        if savefigs: plt.imsave('appearance_prior.png', global_similarity[-continuous_frame].cpu().numpy())
        global_similarity[:-continuous_frame] *= weight_sparse
        # dense spatial prior on recent frames (most recent 4 continuous frames)
        global_similarity[-continuous_frame:] *= weight_dense
        # flow weight on most recent frame
        global_similarity[-1] *= flow_weight
        if savefigs: plt.imsave('final_similarity.png', global_similarity[-continuous_frame].cpu().numpy())
        if savefigs: exit()
    else:
        # dense spatial prior on recent frames (most recent 4 continuous frames)
        global_similarity = global_similarity.mul(weight_dense)

        if savefigs: print('global similarity shape at previous frame: {}'.format(global_similarity[-1].shape))

    global_similarity = global_similarity.view(-1, H * W)

    # get prediction
    prediction = ref_label_selected.mm(global_similarity)

    if savefigs: print('selected ref label shape: {}'.format(ref_label_selected.shape))
    if savefigs: print('prediction shape: {}'.format(prediction.shape))

    return prediction


def sample_frames(frame_idx,
                  take_range,
                  num_refs):
    if frame_idx <= num_refs:
        sample_idx = list(range(frame_idx))
    else:
        dense_num = 4 - 1
        sparse_num = num_refs - dense_num
        target_idx = frame_idx
        ref_end = target_idx - dense_num - 1
        ref_start = max(ref_end - take_range, 0)
        sample_idx = np.linspace(ref_start, ref_end, sparse_num).astype(np.int).tolist()
        for j in range(dense_num):
            sample_idx.append(target_idx - dense_num + j)

    return torch.Tensor(sample_idx).long().cuda()


def prepare_first_frame(curr_video,
                        save_prediction,
                        annotation_dir,
                        sigma1=8,
                        sigma2=21):
    annotation_list = sorted(os.listdir(annotation_dir))
    first_annotation = Image.open(os.path.join(annotation_dir, annotation_list[curr_video], '00000.png'))
    (H, W) = np.asarray(first_annotation).shape
    H_d = int(np.ceil(H / 8))
    W_d = int(np.ceil(W / 8))
    palette = first_annotation.getpalette()
    label = np.asarray(first_annotation)
    d = np.max(label) + 1
    label = torch.Tensor(label).long().cuda()  # (1, H, W)
    label_1hot = idx2onehot(label.view(-1), d).reshape(1, d, H, W)
    label_1hot = torch.nn.functional.interpolate(label_1hot,
                                                           size=(H_d, W_d),
                                                           mode='bilinear',
                                                           align_corners=False)
    label_1hot = label_1hot.reshape(d, -1).unsqueeze(1)
    weight_dense = get_spatial_weight((H_d, W_d), sigma1)
    weight_sparse = get_spatial_weight((H_d, W_d), sigma2)

    if save_prediction is not None:
        if not os.path.exists(save_prediction):
            os.makedirs(save_prediction)
        save_path = os.path.join(save_prediction, annotation_list[curr_video])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        first_annotation.save(os.path.join(save_path, '00000.png'))

    return label_1hot, d, palette, weight_dense, weight_sparse, H, W


def get_spatial_weight(shape, sigma):
    """
    Get soft spatial weights for similarity matrix.
    :param shape: (H, W)
    :param sigma:
    :return: (H*W, H*W)
    """
    (H, W) = shape

    index_matrix = torch.arange(H * W, dtype=torch.long).reshape(H * W, 1).cuda()
    index_matrix = torch.cat((index_matrix / W, index_matrix % W), -1)  # (H*W, 2)
    ##
    if savefigs: print('Spatial prior index matrix shape: {}'.format(index_matrix.shape))
    ##
    d = index_matrix - index_matrix.unsqueeze(1)  # (H*W, H*W, 2)
    if savefigs: plt.imsave('index_matrix.png', index_matrix.cpu().numpy())
    d = d.float().pow(2).sum(-1)  # (H*W, H*W)
    ##
    if savefigs: plt.imsave('unexponentiated_spatial_prior.png', d.cpu().numpy())
    ##
    w = (- d / sigma ** 2).exp()
    ##
    if savefigs: print('Spatial prior shape: {}'.format(w.shape))
    ##

    return w


def get_flow_weight(H, W, flow_history, sigma):
    # optical flow
    curr_flow = cv2.optflow.calcOpticalFlowDenseRLOF(flow_history[1],flow_history[2],None, *[])
    prev_flow = cv2.optflow.calcOpticalFlowDenseRLOF(flow_history[0],flow_history[1],None, *[])

    H_d = int(np.ceil(H / 8))
    W_d = int(np.ceil(W / 8))

    # 8x downsample
    curr_flow = cv2.resize(curr_flow, (H_d, W_d), interpolation=cv2.INTER_LINEAR)
    prev_flow = cv2.resize(prev_flow, (H_d, W_d), interpolation=cv2.INTER_LINEAR)

    # convert to torch tensor
    curr_flow = torch.from_numpy(curr_flow).cuda()
    prev_flow = torch.from_numpy(prev_flow).cuda()

    if savefigs: print('Flow shape: {}'.format(curr_flow.shape))

    curr_flow_0 = curr_flow[:,:,0].squeeze().reshape(H_d*W_d)
    curr_flow_1 = curr_flow[:,:,1].squeeze().reshape(H_d*W_d)

    prev_flow_0 = prev_flow[:,:,0].squeeze().reshape(H_d*W_d, 1)
    prev_flow_1 = prev_flow[:,:,1].squeeze().reshape(H_d*W_d, 1)

    if savefigs: print('Curr flow component shape: {}'.format(curr_flow_0.shape))
    if savefigs: print('Prev flow component shape: {}'.format(prev_flow_0.shape))

    # difference in flows
    diff_flow_0 = curr_flow_0 - prev_flow_0
    diff_flow_1 = curr_flow_1 - prev_flow_1

    if savefigs: print('Flow differential shape: {}'.format(diff_flow_0.shape))

    # squared norm of flow difference
    diff_flow = diff_flow_0.pow(2) + diff_flow_1.pow(2)

    if savefigs: print('Flow differential shape: {}'.format(diff_flow_0.shape))
    if savefigs: plt.imsave('unexponentiated_flow_weight.png', diff_flow.cpu().numpy())

    # exponentiate to convert to a similarity measure
    w = (- diff_flow / sigma ** 2).exp()

    if savefigs: plt.imsave('flow_weight.png', w.cpu().numpy())

    return w
