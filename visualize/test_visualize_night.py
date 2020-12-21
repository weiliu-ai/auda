import numpy as np

import torch.nn.functional as F

from xmuda.data.utils.evaluate import Evaluator
# !/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings

import torch

from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.common.utils.logger import setup_logger
from xmuda.common.utils.metric_logger import MetricLogger
from xmuda.common.utils.torch_util import set_random_seed
from xmuda.models.build import build_model_2d, build_model_3d
from xmuda.data.build import build_dataloader

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from xmuda.data.utils.turbo_cmap import interpolate_or_clip, turbo_colormap_data

# all classes
NUSCENES_COLOR_PALETTE = [
    (255, 158, 0),  # car
    (255, 158, 0),  # truck
    (255, 158, 0),  # bus
    (255, 158, 0),  # trailer
    (255, 158, 0),  # construction_vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # motorcycle
    (255, 61, 99),  # bicycle
    (0, 0, 0),  # traffic_cone
    (0, 0, 0),  # barrier
    (200, 200, 200),  # background
]

# classes after merging (as used in xMUDA)
NUSCENES_COLOR_PALETTE_SHORT = [
    (255, 158, 0),  # vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # bike
    (0, 0, 0),  # traffic boundary
    (200, 200, 200),  # background
]

# all classes
A2D2_COLOR_PALETTE_SHORT = [
    (255, 0, 0),  # car
    (255, 128, 0),  # truck
    (182, 89, 6),  # bike
    (204, 153, 255),  # person
    (255, 0, 255),  # road
    (150, 150, 200),  # parking
    (180, 150, 200),  # sidewalk
    (241, 230, 255),  # building
    (147, 253, 194),  # nature
    (255, 246, 143),  # other-objects
    (0, 0, 0)  # ignore
]

# colors as defined in https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
SEMANTIC_KITTI_ID_TO_BGR = {  # bgr
    0: [0, 0, 0],
    1: [0, 0, 255],
    10: [245, 150, 100],
    11: [245, 230, 100],
    13: [250, 80, 100],
    15: [150, 60, 30],
    16: [255, 0, 0],
    18: [180, 30, 80],
    20: [255, 0, 0],
    30: [30, 30, 255],
    31: [200, 40, 255],
    32: [90, 30, 150],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [75, 0, 175],
    50: [0, 200, 255],
    51: [50, 120, 255],
    52: [0, 150, 255],
    60: [170, 255, 150],
    70: [0, 175, 0],
    71: [0, 60, 135],
    72: [80, 240, 150],
    80: [150, 240, 255],
    81: [0, 0, 255],
    99: [255, 255, 50],
    252: [245, 150, 100],
    256: [255, 0, 0],
    253: [200, 40, 255],
    254: [30, 30, 255],
    255: [90, 30, 150],
    257: [250, 80, 100],
    258: [180, 30, 80],
    259: [255, 0, 0],
}
SEMANTIC_KITTI_COLOR_PALETTE = [SEMANTIC_KITTI_ID_TO_BGR[id] if id in SEMANTIC_KITTI_ID_TO_BGR.keys() else [0, 0, 0]
                                for id in range(list(SEMANTIC_KITTI_ID_TO_BGR.keys())[-1] + 1)]

# classes after merging (as used in xMUDA)
SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR = [
    [245, 150, 100],  # car
    [180, 30, 80],  # truck
    [150, 60, 30],  # bike
    [30, 30, 255],  # person
    [255, 0, 255],  # road
    [255, 150, 255],  # parking
    [75, 0, 75],  # sidewalk
    [0, 200, 255],  # building
    [0, 175, 0],  # nature
    [255, 255, 50],  # other-objects
    [0, 0, 0],  # ignore
]
SEMANTIC_KITTI_COLOR_PALETTE_SHORT = [(c[2], c[1], c[0]) for c in SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR]


def draw_points_image_labels(img, img_indices, seg_labels, show=True, color_palette_type='NuScenes',
                             point_size=0.5, figure_title=None):
    if color_palette_type == 'NuScenes':
        color_palette = NUSCENES_COLOR_PALETTE_SHORT
    elif color_palette_type == 'A2D2':
        color_palette = A2D2_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI_long':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    else:
        raise NotImplementedError('Color palette type not supported')
    color_palette = np.array(color_palette) / 255.
    seg_labels[seg_labels == -100] = len(color_palette) - 1
    colors = color_palette[seg_labels]

    fig = plt.figure()
    plt.imshow(img)
    plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5, s=point_size)
    fig.canvas.set_window_title(figure_title)
    plt.axis('off')
    plt.savefig(figure_title, dpi=300, pad_inches=0.0, bbox_inches='tight')
    plt.pause(1)
    plt.close(fig)


def validate(cfg,
             model_2d,
             model_3d,
             dataloader,
             val_metric_logger,
             pselab_path=None):
    logger = logging.getLogger('xmuda.validate')
    logger.info('Validation')

    # evaluator
    class_names = dataloader.dataset.class_names
    evaluator_2d = Evaluator(class_names)
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_ensemble = Evaluator(class_names) if model_3d else None

    pselab_data_list = []

    end = time.time()
    id_batch = 0
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            id_batch = id_batch + 1
            data_time = time.time() - end
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['seg_label'] = data_batch['seg_label'].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            # predict
            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch) if model_3d else None
            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy()
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            # probs_2d  <class 'torch.Tensor'>, size:torch.Size([19961, 10])
            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1) if model_3d else None
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() if model_3d else None

            # get original point cloud from before voxelization
            seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['orig_points_idx']
            # loop over batch
            left_idx = 0
            # print(type(seg_label))
            # print(seg_label)
            # print(len(seg_label))

            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                assert np.all(curr_points_idx)

                curr_seg_label = seg_label[batch_ind]
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if model_3d else None

                # predicted labels with thresh-moving for 2D + 3D
                beta = 2
                weights_class = np.array([2.68678412, 4.36182969, 5.47896839, 3.89026883, 1.], float)
                m = np.max(weights_class)
                for i in range(np.size(weights_class)):
                    weights_class[i] = ((beta - 1) * weights_class[i] + m - beta) / (m - 1)
                #print(weights_class)
                class_weights = torch.tensor(weights_class.tolist()).cuda()
                probs_2d_tm = probs_2d * class_weights
                probs_3d_tm = probs_3d * class_weights
                pred_label_voxel_ensemble_tm = (probs_2d_tm + probs_3d_tm).argmax(1).cpu().numpy() if model_3d else None
                pred_label_ensemble_tm = pred_label_voxel_ensemble_tm[left_idx:right_idx] if model_3d else None

                # save ground truth and segmentation results into jpg files.
                new_img_PIL = transforms.ToPILImage()(data_batch['img'][batch_ind].cpu()).convert('RGB')
                bg = Image.new('RGB', (new_img_PIL.size[0], new_img_PIL.size[1]), (255, 255, 255))
                # if seg_label[batch_ind].tolist().count(2) > 10 and seg_label[batch_ind].tolist().count(0) > 10:
                # if pred_label_ensemble.tolist().count(2) > 10 and seg_label[batch_ind].tolist().count(2) > 10:
                if (id_batch,batch_ind) in {(14,15), (19,5), (31,4)}:
                    image_name = 'results/night/image_{}_{}.jpg'.format(id_batch, batch_ind)
                    new_img_PIL.save(image_name, quality=95)
                    print(seg_label[batch_ind].tolist().count(2))
                    print(seg_label[batch_ind].tolist().count(0))
                    # print(np.unique(seg_label[batch_ind]))
                    img_indices = data_batch['img_indices'][batch_ind]
                    img_title = 'results/night/image_{}_{}'.format(id_batch, batch_ind)
                    # seg_label_i 如果有行人、vehicle则输出
                    draw_points_image_labels(bg, img_indices, curr_seg_label, color_palette_type='NuScenes',
                                            point_size=1, figure_title=img_title + '_gt.jpg')
                    draw_points_image_labels(bg, img_indices, pred_label_ensemble, color_palette_type='NuScenes',
                                            point_size=1, figure_title=img_title + '_auda.jpg')
                    draw_points_image_labels(bg, img_indices, pred_label_ensemble_tm, color_palette_type='NuScenes',
                                            point_size=1, figure_title=img_title + '_auda_tm.jpg')
                    # draw_points_image_labels(bg, img_indices, pred_label_ensemble, color_palette_type='NuScenes',
                    #                          point_size=1, figure_title=img_title + '_xmuda.jpg')

                # evaluate
                evaluator_2d.update(pred_label_2d, curr_seg_label)
                if model_3d:
                    evaluator_3d.update(pred_label_3d, curr_seg_label)
                    evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)

                if pselab_path is not None:
                    assert np.all(pred_label_2d >= 0)
                    curr_probs_2d = probs_2d[left_idx:right_idx]
                    curr_probs_3d = probs_3d[left_idx:right_idx] if model_3d else None
                    pselab_data_list.append({
                        'probs_2d': curr_probs_2d[range(len(pred_label_2d)), pred_label_2d].cpu().numpy(),
                        'pseudo_label_2d': pred_label_2d.astype(np.uint8),
                        'probs_3d': curr_probs_3d[
                            range(len(pred_label_3d)), pred_label_3d].cpu().numpy() if model_3d else None,
                        'pseudo_label_3d': pred_label_3d.astype(np.uint8) if model_3d else None
                    })

                left_idx = right_idx

            seg_loss_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch['seg_label'])
            seg_loss_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch['seg_label']) if model_3d else None
            val_metric_logger.update(seg_loss_2d=seg_loss_2d)
            if seg_loss_3d is not None:
                val_metric_logger.update(seg_loss_3d=seg_loss_3d)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        val_metric_logger.update(seg_iou_2d=evaluator_2d.overall_iou)
        if evaluator_3d is not None:
            val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
        eval_list = [('2D', evaluator_2d)]
        if model_3d:
            eval_list.extend([('3D', evaluator_3d), ('2D+3D', evaluator_ensemble)])
        for modality, evaluator in eval_list:
            np.set_printoptions(suppress=True)
            logger.info('{} overall accuracy={:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU={:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} Average precision ={:.2f}'.format(modality, 100.0 * evaluator.average_precision))
            # logger.info('{} confusion matrix : \n{}'.format(modality, evaluator.confusion_matrix ))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))

        if pselab_path is not None:
            np.save(pselab_path, pselab_data_list)
            logger.info('Saved pseudo label data to {}'.format(pselab_path))


def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA test')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument('ckpt2d', type=str, help='path to checkpoint file of the 2D model')
    parser.add_argument('ckpt3d', type=str, help='path to checkpoint file of the 3D model')
    parser.add_argument('--pselab', action='store_true', help='generate pseudo-labels')
    parser.add_argument('--threshmoving', action='store_true', default=False, help='thresh moving')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def test(cfg, args, output_dir=''):
    logger = logging.getLogger('xmuda.test')

    # build 2d model
    model_2d = build_model_2d(cfg)[0]

    # build 3d model
    model_3d = build_model_3d(cfg)[0]

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()

    # build checkpointer
    checkpointer_2d = CheckpointerV2(model_2d, save_dir=output_dir, logger=logger)
    if args.ckpt2d:
        # load weight if specified
        weight_path = args.ckpt2d.replace('@', output_dir)
        checkpointer_2d.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer_2d.load(None, resume=True)
    checkpointer_3d = CheckpointerV2(model_3d, save_dir=output_dir, logger=logger)
    if args.ckpt3d:
        # load weight if specified
        weight_path = args.ckpt3d.replace('@', output_dir)
        checkpointer_3d.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer_3d.load(None, resume=True)

    # build dataset
    test_dataloader = build_dataloader(cfg, mode='test', domain='target')

    pselab_path = None
    if args.pselab:
        pselab_dir = osp.join(output_dir, 'pselab_data')
        os.makedirs(pselab_dir, exist_ok=True)
        assert len(cfg.DATASET_TARGET.TEST) == 1
        pselab_path = osp.join(pselab_dir, cfg.DATASET_TARGET.TEST[0] + '.npy')

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #

    set_random_seed(cfg.RNG_SEED)
    test_metric_logger = MetricLogger(delimiter='  ')
    model_2d.eval()
    model_3d.eval()

    validate(cfg, model_2d, model_3d, test_dataloader, test_metric_logger, pselab_path=pselab_path)


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from xmuda.common.config import purge_cfg
    from xmuda.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('xmuda', output_dir, comment='test.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    test(cfg, args, output_dir)


if __name__ == '__main__':
    main()
