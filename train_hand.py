import logging
import os
from pathlib import Path

import yaml
import time
import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from torch.utils.tensorboard import SummaryWriter
from ptsemseg.checkpoints import load_model_state, resolve_checkpoint_path, resolve_device
from ptsemseg.models.utils import MergeParametric
from utils import train_parser, validate_parser, RNG_SEED
from validate import final_prediction_output, validate, wrap_str


MODULE_LOGGER = logging.getLogger("ptsemseg")


def best_model_path(cfg):
    return "{}_{}_best_model.pkl".format(
        cfg['model']['arch'],
        cfg['data']['dataset'])


def resolve_training_checkpoint(cfg, logger):
    resume_value = cfg.get("training", {}).get("resume")
    if not resume_value:
        return None

    try:
        return resolve_checkpoint_path(resume=resume_value, logdir=cfg.get("logdir"))
    except FileNotFoundError as exc:
        logger.warning(str(exc))
    return None


def overwrite(cfg, args):

    if args.scale_weight >= 0.:
        cfg['training']['loss']['scale_weight'] = args.scale_weight
    if args.model != "":
        cfg['model']['arch'] = args.model
    # update the model parameters in this config.
    for name in ["initial", "steps", "gate", "hidden_size", "feature_scale", "structure"]:
        value = getattr(args, name)
        if value is not None:
            cfg['model'][name] = value
    if args.batch_size != 0:
        cfg['training']['batch_size'] = args.batch_size
    if args.lr_n != 0:
        cfg['training']['optimizer']['lr'] = args.lr_n*1.0/(10**args.lr_exponent)

    # These two only operates on R-UNet.
    if cfg['model']['arch'] == "runet":
        for name in ["unet_level", "recurrent_level"]:
            value = getattr(args, name)
            if value is not None:
                cfg['model'][name] = value
    cfg['training']['prefix'] = args.prefix

    """ Change the valid steps, if baseline model """
    if args.prefix == 'baseline':
        cfg['training']['val_interval'] = 6000
        cfg['training']['print_interval'] = 500

    if args.prefix == 'benchmark':
        cfg['training']['val_interval'] = 10
        cfg['training']['train_iters'] = 11
        cfg['training']['print_interval'] = 10

    if args.loss is not None:
        cfg['training']['loss']['name'] = args.loss

    if cfg['training']['loss']['name'] == 'cross_entropy':
        cfg['training']['loss'].pop('scale_weight', None)

    # manipulate learning rate
    args.lr = float(args.lr)
    cfg['training']['optimizer']['lr'] = float(cfg['training']['optimizer']['lr'])
    print("args.lr is {}".format(args.lr))
    if 1 > float(args.lr) > 0:
        # Only override if lr is given positive value from (0,1)
        cfg['training']['optimizer']['lr'] = args.lr

    # Over write the void class
    cfg['data'].setdefault('void_class', -1)
    return cfg


def normalize_args_from_cfg(args, cfg):
    model_defaults = {
        "steps": 3,
        "hidden_size": 32,
        "initial": 1,
        "gate": 2,
        "feature_scale": 4,
        "unet_level": 4,
        "recurrent_level": -1,
        "structure": "ours",
    }
    for name, default in model_defaults.items():
        value = cfg.get("model", {}).get(name, default)
        cfg.setdefault("model", {})[name] = value
        setattr(args, name, value)
    if args.clip is None:
        args.clip = 10.0
    args.batch_size = cfg["training"]["batch_size"]
    args.loss = cfg["training"]["loss"]["name"]
    return args


def load_cfg_with_overwrite(args):
    # Overwrite the figure.
    with open(args.config, encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    cfg = overwrite(cfg, args)
    normalize_args_from_cfg(args, cfg)

    cfg['run_id'] = run_id = random.randint(1, 100000)
    config_name = os.path.basename(args.config)[:-4]
    config_name = args.model + '_' + config_name if len(args.model) > 0 else config_name

    if use_grad_clip(cfg['model']['arch']) and cfg['model']['arch'] != 'unet':
        logdir = os.path.join('runs', cfg['data']['dataset'], cfg['training']['prefix'],
                              '{}-h{}-{}-r{}-w-{}-gate{}-bs-{}-fscale-{}'.format(
                                  config_name,
                                  args.hidden_size,
                                  args.initial,
                                  args.steps,
                                  cfg['training']['loss']['scale_weight'],
                                  args.gate,
                                  args.batch_size,
                                  args.feature_scale,
                              ),
                              str(run_id))

    if cfg['model']['arch'] in ['vanillarnnunet', 'unethidden', 'vanillarnnunetr']:
        logdir = os.path.join('runs', cfg['data']['dataset'], cfg['training']['prefix'],
                              '{}-h{}-w-{}-bs-{}-fscale-{}'.format(
                                  config_name,
                                  args.initial,
                                  cfg['training']['loss']['scale_weight'],
                                  args.batch_size,
                                  args.feature_scale,
                              ),
                              str(run_id))

    elif cfg['model']['arch'] in ['unet', 'unet_expand', 'unet_expand_all', 'unetbnslim', 'unetgnslim',
                                  'unet_deep_as_dru', 'deeplabv3']:
        logdir = os.path.join('runs', cfg['data']['dataset'], cfg['training']['prefix'],
                              '{}-bs-{}-fscale-{}'.format(
                                  config_name,
                                  args.batch_size,
                                  args.feature_scale,
                              ),
                              str(run_id))

    elif cfg['model']['arch'] in ['unetvgg11', 'unetvgg16', 'unetvgg16gn', 'unetresnet50', 'unetresnet50bn']:
        logdir = os.path.join('runs', cfg['data']['dataset'], cfg['training']['prefix'],
                              '{}-bs-{}'.format(
                                  config_name,
                                  args.batch_size,
                              ),
                              str(run_id))

    else:
        logdir = os.path.join('runs', cfg['data']['dataset'], cfg['training']['prefix'],
                              '{}-h{}-{}-r{}-gate{}-bs-{}-fscale-{}'.format(
                                  config_name,
                                  args.hidden_size,
                                  args.initial,
                                  args.steps,
                                  args.gate,
                                  args.batch_size,
                                  args.feature_scale,
                              ),
                              str(run_id))

    cfg['config'] = config_name
    cfg['logdir'] = logdir
    cfg["training"].setdefault("resume", None)

    return cfg


def use_grad_clip(name):
    if 'rcnn' in name:
        return True
    if 'runet' in name:
        return True
    # if "gruunetnew" == name:
    #     return True
    if "gruunet" in name:
        return True
    if "NoParamShare" in name:
        return True
    if "rnnunet" in name:
        return True
    if "rec" in name:
        return True
    if "dru" in name:
        return True
    if "sru" in name:
        return True
    return False


def weights_init(m, init_logger=None):
    init_logger = init_logger or MODULE_LOGGER
    if isinstance(m, MergeParametric):
        init_logger.info('initializing merge layer ...')
        pass
    elif isinstance(m, nn.Conv2d):
        init_logger.warning(f'initializing {m}')
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        init_logger.warning(f'initializing {m}')
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_model(model, init_logger=None):
    init_logger = init_logger or MODULE_LOGGER
    for m in model.paramGroup2.modules():
        if isinstance(m, nn.GroupNorm) or isinstance(m, _BatchNorm):
            init_logger.warning(f'initializing {m}')
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        else:
            weights_init(m, init_logger)


def train(cfg, writer, logger, args):

    # Setup seeds
    torch.manual_seed(cfg.get('seed', RNG_SEED))
    np.random.seed(cfg.get('seed', RNG_SEED))
    random.seed(cfg.get('seed', RNG_SEED))

    # Setup device
    device = resolve_device(args.device)
    logger.info("Using device: %s", device)
    logger.info("Using config: %s", args.config)

    # Setup Augmentations
    # augmentations = cfg['training'].get('augmentations', None)
    if cfg['data']['dataset'] in ['drive']:
        augmentations = cfg['training'].get('augmentations',
                                            {'brightness': 63. / 255.,
                                             'saturation': 0.5,
                                             'contrast': 0.8,
                                             'hflip': 0.5,
                                             'rotate': 180,
                                             'rscalecropsquare': 576,
                                             })
        # augmentations = cfg['training'].get('augmentations',
        #                                     {'rotate': 10, 'hflip': 0.5, 'rscalecrop': 512, 'gaussian': 0.5})
    else:
        augmentations = cfg['training'].get('augmentations', {'rotate': 10, 'hflip': 0.5})
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']
    logger.info("Dataset path: %s", data_path)

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug)

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),)

    n_classes = t_loader.n_classes
    n_workers = int(cfg['training'].get('n_workers', 0))
    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'],
                                  num_workers=n_workers,
                                  shuffle=True)

    valloader = data.DataLoader(v_loader,
                                batch_size=cfg['training']['batch_size'],
                                num_workers=n_workers)

    # Setup Metrics
    running_metrics_val = runningScore(n_classes, cfg['data']['void_class'] > 0)

    # Setup Model
    print('trying device {}'.format(device))
    model = get_model(cfg['model'], n_classes, args).to(device)
    logger.info("Using model: %s", cfg['model']['arch'])

    if cfg['model']['arch'] not in ['unetvgg16', 'unetvgg16gn', 'druvgg16', 'unetresnet50', 'unetresnet50bn',
                                    'druresnet50', 'druresnet50bn']:
        model.apply(lambda module: weights_init(module, logger))
    else:
        init_model(model, logger)

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items()
                        if k != 'name'}
    if cfg['model']['arch'] in ['unetvgg16', 'unetvgg16gn', 'druvgg16', 'druresnet50', 'druresnet50bn']:
        optimizer = optimizer_cls([
            {'params': model.paramGroup1.parameters(), 'lr': optimizer_params['lr'] / 10},
            {'params': model.paramGroup2.parameters()}
        ], **optimizer_params)
    else:
        optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.warning(f"Model parameters in total: {sum([p.numel() for p in model.parameters()])}")
    logger.warning(f"Trainable parameters in total: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    best_iou = -100.0
    resume_path = resolve_training_checkpoint(cfg, logger)
    best_checkpoint_path = resume_path
    if resume_path is not None:
        logger.info("Loading checkpoint: %s", resume_path)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(load_model_state(resume_path, device))
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_iter = int(checkpoint.get("epoch", 0))
        best_iou = float(checkpoint.get("best_iou", best_iou))
        logger.info(
            "Loaded checkpoint %s at iteration %d with best IoU %.6f",
            resume_path,
            start_iter,
            best_iou,
        )

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    i = start_iter
    flag = True

    weight = torch.ones(n_classes)
    if cfg['data'].get('void_class'):
        if cfg['data'].get('void_class') >= 0:
            weight[cfg['data'].get('void_class')] = 0.
    weight = weight.to(device)

    logger.info("Set the prediction weights as {}".format(weight))

    while i < cfg['training']['train_iters'] and flag:
        for batch in trainloader:
            images, labels = batch[:2]
            i += 1
            start_ts = time.time()
            model.train()
            # for param_group in optimizer.param_groups:
            #     print(param_group['lr'])
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            if cfg['model']['arch'] in ['reclast']:
                h0 = images.new_ones([images.shape[0], args.hidden_size, images.shape[2], images.shape[3]])
                outputs = model(images, h0)

            elif cfg['model']['arch'] in ['recmid']:
                W, H = images.shape[2], images.shape[3]
                w = int(np.floor(np.floor(np.floor(W/2)/2)/2)/2)
                h = int(np.floor(np.floor(np.floor(H/2)/2)/2)/2)
                h0 = images.new_ones([images.shape[0], args.hidden_size, w, h])
                outputs = model(images, h0)

            elif cfg['model']['arch'] in ['dru', 'sru']:
                W, H = images.shape[2], images.shape[3]
                w = int(np.floor(np.floor(np.floor(W/2)/2)/2)/2)
                h = int(np.floor(np.floor(np.floor(H/2)/2)/2)/2)
                h0 = images.new_ones([images.shape[0], args.hidden_size, w, h])
                s0 = images.new_ones([images.shape[0], n_classes, W, H])
                outputs = model(images, h0, s0)

            elif cfg['model']['arch'] in ['druvgg16', 'druresnet50', 'druresnet50bn']:
                W, H = images.shape[2], images.shape[3]
                w, h = int(W / 2 ** 4), int(H / 2 ** 4)
                if cfg['model']['arch'] in ['druresnet50', 'druresnet50bn']:
                    w, h = int(W / 2 ** 5), int(H / 2 ** 5)
                h0 = images.new_ones([images.shape[0], args.hidden_size, w, h])
                s0 = images.new_zeros([images.shape[0], n_classes, W, H])
                outputs = model(images, h0, s0)

            else:
                outputs = model(images)

            loss = loss_fn(input=outputs, target=labels, weight=weight, bkargs=args)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # if use_grad_clip(cfg['model']['arch']):  #
            # if cfg['model']['arch'] in ['rcnn', 'rcnn2', 'rcnn3']:  #
            if use_grad_clip(cfg['model']['arch']):
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()
            scheduler.step()

            time_meter.update(time.time() - start_ts)

            if i % cfg['training']['print_interval'] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(i,
                                           cfg['training']['train_iters'], 
                                           loss.item(),
                                           time_meter.avg / cfg['training']['batch_size'])

                # print(print_str)
                logger.info(print_str)
                writer.add_scalar('loss/train_loss', loss.item(), i)
                time_meter.reset()

            if i % cfg['training']['val_interval'] == 0 or \
               i == cfg['training']['train_iters']:
                model.eval()
                with torch.no_grad():
                    for i_val, batch_val in tqdm(enumerate(valloader)):
                        images_val, labels_val = batch_val[:2]
                        if args.benchmark:
                            if i_val > 10:
                                break
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)
                        if cfg['model']['arch'] in ['reclast']:
                            h0 = images_val.new_ones([images_val.shape[0], args.hidden_size, images_val.shape[2], images_val.shape[3]])
                            outputs = model(images_val, h0)

                        elif cfg['model']['arch'] in ['recmid']:
                            W, H = images_val.shape[2], images_val.shape[3]
                            w = int(np.floor(np.floor(np.floor(W / 2) / 2) / 2) / 2)
                            h = int(np.floor(np.floor(np.floor(H / 2) / 2) / 2) / 2)
                            h0 = images_val.new_ones([images_val.shape[0], args.hidden_size, w, h])
                            outputs = model(images_val, h0)

                        elif cfg['model']['arch'] in ['dru', 'sru']:
                            W, H = images_val.shape[2], images_val.shape[3]
                            w = int(np.floor(np.floor(np.floor(W / 2) / 2) / 2) / 2)
                            h = int(np.floor(np.floor(np.floor(H / 2) / 2) / 2) / 2)
                            h0 = images_val.new_ones([images_val.shape[0], args.hidden_size, w, h])
                            s0 = images_val.new_ones([images_val.shape[0], n_classes, W, H])
                            outputs = model(images_val, h0, s0)

                        elif cfg['model']['arch'] in ['druvgg16', 'druresnet50', 'druresnet50bn']:
                            W, H = images_val.shape[2], images_val.shape[3]
                            w, h = int(W / 2**4), int(H / 2**4)
                            if cfg['model']['arch'] in ['druresnet50', 'druresnet50bn']:
                                w, h = int(W / 2 ** 5), int(H / 2 ** 5)
                            h0 = images_val.new_ones([images_val.shape[0], args.hidden_size, w, h])
                            s0 = images_val.new_zeros([images_val.shape[0], n_classes, W, H])
                            outputs = model(images_val, h0, s0)

                        else:
                            outputs = model(images_val)
                        val_loss = loss_fn(input=outputs, target=labels_val, weight=weight, bkargs=args)

                        pred = final_prediction_output(outputs).detach().argmax(1).cpu().numpy()
                        gt = labels_val.detach().cpu().numpy()
                        logger.debug('pred shape: %s\t ground-truth shape: %s', pred.shape, gt.shape)
                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())
                    # assert i_val > 0, "Validation dataset is empty for no reason."
                writer.add_scalar('loss/val_loss', val_loss_meter.avg, i)
                logger.info("Iter %d Loss: %.4f" % (i, val_loss_meter.avg))
                score, class_iou, _ = running_metrics_val.get_scores()
                for k, v in score.items():
                    # print(k, v)
                    logger.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/{}'.format(k), v, i)

                for k, v in class_iou.items():
                    logger.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/cls_{}'.format(k), v, i)

                val_loss_meter.reset()
                running_metrics_val.reset()

                mean_iou = float(score["Mean IoU : \t"])
                if mean_iou >= best_iou:
                    best_iou = mean_iou
                    state = {
                        "epoch": i,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = Path(writer.file_writer.get_logdir()) / best_model_path(cfg)
                    torch.save(state, save_path)
                    best_checkpoint_path = save_path

            if i == cfg['training']['train_iters']:
                flag = False
                save_path = os.path.join(writer.file_writer.get_logdir(),
                                         "{}_{}_final_model.pkl".format(
                                             cfg['model']['arch'],
                                             cfg['data']['dataset']))
                final_state = {
                    "epoch": i,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_iou": best_iou,
                }
                torch.save(final_state, save_path)
                break

    return best_checkpoint_path


if __name__ == "__main__":

    parser = train_parser()
    args = parser.parse_args()

    cfg = load_cfg_with_overwrite(args)

    run_id = cfg['run_id']
    logdir = cfg['logdir']

    writer = SummaryWriter(log_dir=logdir)

    with open(os.path.join(logdir, 'config.yaml'), 'w', encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp, default_flow_style=False)

    print('RUNDIR: {}'.format(logdir))

    # Write the config file to logdir
    # shutil.copy(args.config, logdir)

    logger = get_logger(logdir, level=logging.WARN if args.prefix == 'benchmark' else logging.INFO)
    logger.info('Let the games begin')

    best_checkpoint_path = None
    try:
        best_checkpoint_path = train(cfg, writer, logger, args)
    # except (RuntimeError, KeyboardInterrupt) as e:
    except (KeyboardInterrupt) as e:
        logger.error(e)
    finally:
        writer.close()

    logger.info("\nValidate the training result...")
    valid_parser = validate_parser(parser)
    valid_args = valid_parser.parse_args()
    # set the model path.
    # valid_args.steps = 3
    if args.prefix == 'benchmark':
        valid_args.benchmark = True
    if best_checkpoint_path is not None and Path(best_checkpoint_path).is_file():
        valid_args.model_path = str(best_checkpoint_path)
        validate(cfg, valid_args)
    else:
        logger.warning("Skipping final validation because no best checkpoint was produced.")

# --config=configs/dataset/eythhand.yml --model=unetvgg16 --lr=1e-8 /
# --batch_size=8 --structure=unetvgg16 --loss=cross_entropy --prefix=iccvablation

# python train_hand.py --config=configs/dataset/eythhand.yml --model=unetvgg16gn --lr=1e-8 /
# --batch_size=8 --structure=unetvgg16gn --loss=cross_entropy --prefix=iccvablation

# --config=configs/dataset/eythhand.yml --model=unetresnet50 --lr=1e-8 --batch_size=8 --structure=unetresnet50 --loss=cross_entropy --prefix=iccvablation
