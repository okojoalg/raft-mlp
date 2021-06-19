from datetime import datetime
from pathlib import Path

import ignite
import ignite.distributed as idist
import torch
import torch.optim as optim
from ignite.contrib.engines import common
from ignite.contrib.handlers import ClearMLLogger, PiecewiseLinear
from ignite.contrib.handlers.clearml_logger import ClearMLSaver
from ignite.engine import Engine, Events
from ignite.handlers import DiskSaver, Checkpoint, global_step_from_engine
from ignite.metrics import Loss, Accuracy, TopKCategoricalAccuracy
from ignite.utils import manual_seed, setup_logger
from omegaconf import open_dict
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from libs.const import CIFAR10, CIFAR100, IMAGENET, LOGGER_NAME
from libs.datasets import ImageNetGetter, CIFAR10Getter, CIFAR100Getter
from libs.models import PyramidMixer


def training(local_rank, params):
    rank = idist.get_rank()
    manual_seed(params.settings.seed + rank)
    device = idist.device()

    logger = setup_logger(name=LOGGER_NAME, distributed_rank=local_rank)
    clearml_logger = ClearMLLogger(
        project_name=params.settings.project_name,
        task_name=params.settings.task_name,
    ) if params.settings.with_clearml else None

    log_basic_info(logger, params)

    if rank == 0:
        if params.settings.stop_iteration is None:
            now = datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            now = f"stop-on-{params.settings.stop_iteration}"

        folder_name = f"{idist.get_world_size()}_{now}"
        output_path = Path(params.settings.project_name) / params.settings.task_name / folder_name
        with open_dict(params):
            params.settings.output_path = f"s3://{output_path.as_posix()}"
        logger.info(f"Output path: {params.settings.output_path}")

        if "cuda" in device.type:
            with open_dict(params):
                params.settings.cuda_device_name = torch.cuda.get_device_name(local_rank)

        if params.settings.with_clearml:
            try:
                from clearml import Task
            except ImportError:
                from trains import Task

            task = Task.init(project_name=params.settings.project_name, task_name=params.settings.task_name)
            task.connect_configuration(dict(params.settings))
            hyper_params = [
                "batch_size",
                "weight_decay",
                "num_epochs",
                "lr",
                "num_warmup_epochs",
            ]
            task.connect({k: params.settings[k] for k in hyper_params})

    train_loader, val_loader, num_classes, image_size, channels = get_dataflow(params)

    with open_dict(params):
        params.settings.num_iters_per_epoch = len(train_loader)
        params.settings.num_classes = num_classes
        params.settings.image_size = image_size
        params.settings.channels = channels
    model, optimizer, criterion, lr_scheduler = initialize(params)

    trainer = create_trainer(model, optimizer, criterion, lr_scheduler,
                             train_loader.sampler, params, logger, clearml_logger)

    metrics = {
        "accuracy": Accuracy(),
        "top5-accuracy": TopKCategoricalAccuracy(k=5),
        "loss": Loss(criterion),
    }

    evaluator = create_evaluator(model, metrics=metrics, params=params)
    train_evaluator = create_evaluator(model, metrics=metrics, params=params)

    def run_validation(engine):
        epoch = trainer.state.epoch
        state = train_evaluator.run(train_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Train", state.metrics)
        state = evaluator.run(val_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Test", state.metrics)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=params.settings.validate_every) | Events.COMPLETED, run_validation)
    if params.settings.with_clearml:
        clearml_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag='training/loss',
            metric_names=['loss'],
            global_step_transform=global_step_from_engine(trainer, Events.ITERATION_COMPLETED),
        )
        clearml_logger.attach_output_handler(
            train_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag='train/accuracy',
            metric_names=['accuracy', 'top5-accuracy'],
            global_step_transform=global_step_from_engine(trainer, Events.EPOCH_COMPLETED),
        )
        clearml_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag='validation/loss',
            metric_names=['loss'],
            global_step_transform=global_step_from_engine(trainer, Events.EPOCH_COMPLETED),
        )
        clearml_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag='validation/accuracy',
            metric_names=['accuracy', 'top5-accuracy'],
            global_step_transform=global_step_from_engine(trainer, Events.EPOCH_COMPLETED),
        )
        clearml_logger.attach_opt_params_handler(trainer, event_name=Events.ITERATION_STARTED, optimizer=optimizer, param_name='lr')

    best_model_handler = Checkpoint(
        {"model": model},
        get_save_handler(params, clearml_logger),
        filename_prefix="best",
        n_saved=2,
        global_step_transform=global_step_from_engine(trainer),
        score_name="val_accuracy",
        score_function=Checkpoint.get_default_score_fn("accuracy"),
    )
    evaluator.add_event_handler(
        Events.COMPLETED(lambda *_: trainer.state.epoch > params.settings.num_epochs // 2), best_model_handler
    )

    if params.settings.stop_iteration is not None:
        @trainer.on(Events.ITERATION_STARTED(once=params.settings.stop_iteration))
        def _():
            logger.info(f"Stop training on {trainer.state.iteration} iteration")
            trainer.terminate()

    try:
        trainer.run(train_loader, max_epochs=params.settings.num_epochs)
    except Exception as e:
        logger.exception("")
        raise e


def get_dataflow(params):
    if idist.get_local_rank() > 0:
        idist.barrier()

    if params.settings.dataset_name == IMAGENET:
        dg = ImageNetGetter()
    elif params.settings.dataset_name == CIFAR10:
        dg = CIFAR10Getter()
    elif params.settings.dataset_name == CIFAR100:
        dg = CIFAR100Getter()
    else:
        raise ValueError("Invalid dataset name")
    train_ds, val_ds = dg.get(params.settings.data_path)

    if idist.get_local_rank() == 0:
        idist.barrier()

    train_loader = idist.auto_dataloader(
        train_ds, batch_size=params.settings.batch_size, num_workers=params.settings.num_workers, shuffle=True, drop_last=True,
    )

    val_loader = idist.auto_dataloader(
        val_ds, batch_size=2 * params.settings.batch_size, num_workers=params.settings.num_workers, shuffle=False,
    )
    return train_loader, val_loader, dg.num_classes, dg.image_size, dg.channels


def initialize(params):
    model = PyramidMixer(
        layers=params.settings.layers,
        in_channels=params.settings.channels,
        image_size=params.settings.image_size,
        num_classes=params.settings.num_classes,
        expansion_factor=params.settings.expansion_factor,
        dropout=params.settings.dropout,
    )
    model = idist.auto_model(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=params.settings.lr,
        weight_decay=params.settings.weight_decay,
    )
    optimizer = idist.auto_optim(optimizer)
    criterion = nn.CrossEntropyLoss().to(idist.device())

    le = params.settings.num_iters_per_epoch
    milestones_values = [
        (0, 0.0),
        (le * params.settings.num_warmup_epochs, params.settings.lr),
        (le * params.settings.num_epochs, 0.0),
    ]
    lr_scheduler = PiecewiseLinear(optimizer, param_name="lr", milestones_values=milestones_values)

    return model, optimizer, criterion, lr_scheduler


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}")


def log_basic_info(logger, params):
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        from torch.backends import cudnn

        logger.info(f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in params.settings.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


def create_trainer(model, optimizer, criterion, lr_scheduler, train_sampler, params, logger, clearml_logger):
    device = idist.device()
    with_amp = params.settings.with_amp
    scaler = GradScaler(enabled=with_amp)

    def train_step(engine, batch):

        x, y = batch[0], batch[1]

        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        model.train()

        with autocast(enabled=with_amp):
            y_pred = model(x)
            loss = criterion(y_pred, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return {
            "loss": loss.item(),
        }

    trainer = Engine(train_step)
    trainer.logger = logger

    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
    metric_names = [
        "loss",
    ]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=params.settings.checkpoint_every,
        save_handler=get_save_handler(params, clearml_logger),
        lr_scheduler=lr_scheduler,
        output_names=metric_names if params.settings.log_every_iters > 0 else None,
        with_pbars=False,
        clear_cuda_cache=False,
    )

    resume_from = params.settings.resume_from
    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
        logger.info(f"Resume from a checkpoint: {checkpoint_fp.as_posix()}")
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer


def create_evaluator(model, metrics, params, tag="val"):
    with_amp = params.settings.with_amp
    device = idist.device()

    @torch.no_grad()
    def evaluate_step(engine: Engine, batch):
        model.eval()
        x, y = batch[0], batch[1]
        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        with autocast(enabled=with_amp):
            output = model(x)
        return output, y

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    if idist.get_rank() == 0 and (not params.settings.with_clearml):
        common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(evaluator)

    return evaluator


def get_save_handler(params, clearml_logger):
    if params.settings.with_clearml:
        return ClearMLSaver(logger=clearml_logger, output_uri=params.settings.output_path)

    return DiskSaver(params.settings.output_path, require_empty=False)


def main(params, **kwargs):
    kwargs["nproc_per_node"] = params.settings.nproc_per_node
    if params.settings.backend == "xla-tpu" and params.settings.with_amp:
        raise RuntimeError("The value of with_amp should be False if backend is xla")

    with idist.Parallel(backend=params.settings.backend, **kwargs) as parallel:
        parallel.run(training, params)
