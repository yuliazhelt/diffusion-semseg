# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
from datetime import datetime

import wandb 
import mmcv
import torch
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.apis import multi_gpu_test, single_gpu_test, inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes)
from mmseg.utils import build_ddp, build_dp, setup_multi_processes

import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # distributed training
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')

    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    args.eval = "mIoU"
    args.aug_test = False # singe scale mIoU
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.work_dir = args.work_dir
    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = os.path.join('./work_dirs',
                                os.path.splitext(os.path.basename(args.config))[0])

    
    cfg.img_dir = os.path.join(cfg.work_dir, "val_images")

    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    mmcv.mkdir_or_exist(os.path.abspath(cfg.img_dir))
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))

    setup_multi_processes(cfg)

    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.launcher == 'none':
        distributed = False
        cfg.gpu_ids = range(1)
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    cfg.device = get_device()

    meta = dict()
    seed = init_random_seed(args.seed, device=cfg.device)
    set_random_seed(seed)
    cfg.seed = seed
    meta['seed'] = seed

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    model.init_weights()

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))


    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE
        )
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    if cfg.checkpoint_config is not None:
        meta.update(cfg.checkpoint_config.meta)


    timestamp = datetime.now().strftime('%H-%M-%S_%Y-%m-%d')
    run_name = f"{cfg.model.type}_{model.caption_type}_train_{timestamp}"
    cfg.log_config.hooks[0]['init_kwargs']['name'] = run_name
    cfg.log_config.hooks[0]['init_kwargs']['config'] = cfg


    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        timestamp=timestamp,
        meta=meta
    )

    wandb.init(project='thesis_vpd', name=f"test_{run_name}", config=cfg)


    if args.aug_test:
        # hard code index
        # cfg.data.test.pipeline[1].img_ratios = [
        #     0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        # ]
        cfg.data.test.pipeline[1].img_ratios = [
            1.0, 1.125, 1.25, 1.375, 1.5
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
 
    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(args.work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(args.work_dir,
                                 f'eval_single_scale_{timestamp}.json')
    elif rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(work_dir,
                                 f'eval_single_scale_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # The default loader config
    loader_cfg = dict(
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None

    model.CLASSES = dataset.CLASSES
    model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {}

    eval_on_format_results = (
        args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    cfg.device = get_device()
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        model = revert_sync_batchnorm(model)
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        results = single_gpu_test(
            model,
            data_loader,
            args.show,
            args.show_dir,
            False,
            args.opacity,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)

        model.cfg = cfg

        # # log to wandb      
        # with torch.no_grad():
        #     for i, batch in enumerate(data_loader):
        #         # batch_size=1, log 30 images
        #         if i > 30:
        #             break
        #         img_metas = batch['img_metas'][0].data[0]
        #         filename = img_metas[0]['filename']

        #         img = mmcv.imread(filename)
        #         result = inference_segmentor(model, img)
        #         show_result_pyplot(model, img, result, out_file=f'{cfg.img_dir}/val_{i}.png')
        #         # Log the saved image to wandb
        #         wandb.log({"Segmentation Result": [wandb.Image(f'{cfg.img_dir}/val_{i}.png', caption=f"result_{filename.split('/')[-1]}")]})

        results = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            False,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs
        )

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            warnings.warn(
                'The behavior of ``args.out`` has been changed since MMSeg '
                'v0.16, the pickled outputs could be seg map as type of '
                'np.array, pre-eval results or file paths for '
                '``dataset.format_results()``.')
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
        if args.eval:
            eval_kwargs.update(metric=args.eval)
            metric = dataset.evaluate(results, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            mmcv.dump(metric_dict, json_file, indent=4)
            if tmpdir is not None and eval_on_format_results:
                # remove tmp dir when cityscapes evaluation
                shutil.rmtree(tmpdir)

            per_class_columns = ['Class', 'IoU', 'Acc']
            per_class_data = []

            for key, value in metric_dict['metric'].items():
                if key.startswith('IoU.'):
                    class_name = key[4:]  # Remove 'IoU.' prefix
                    iou = value
                    acc = metric_dict['metric'].get(f'Acc.{class_name}', None)
                    per_class_data.append([class_name, iou * 100, acc * 100])  # Convert fractions to percentages

            per_class_table = wandb.Table(columns=per_class_columns, data=per_class_data)
            wandb.log({"Per-Class Results": per_class_table})

            wandb_columns = ['Metric', 'Value']
            wandb_data = [
                ['aAcc', metric_dict['metric']['aAcc'] * 100],  # Convert fractions to percentages
                ['mIoU', metric_dict['metric']['mIoU'] * 100],
                ['mAcc', metric_dict['metric']['mAcc'] * 100]
            ]

            wandb_table = wandb.Table(columns=wandb_columns, data=wandb_data)
            wandb.log({"Summary Metrics": wandb_table})


if __name__ == '__main__':
    main()
