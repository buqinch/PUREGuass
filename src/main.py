import os
from pathlib import Path

import hydra
import torch
import wandb
import signal
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf


from src.misc.weight_modify import checkpoint_filter_fn
from src.model.distiller import get_distiller

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

class LossLoggerCallback(Callback):
    """
    Callback that logs training loss to a text file at each training step.
    """
    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'w') as f:
            f.write("step\tloss\n")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = None
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
        else:
            loss = trainer.callback_metrics.get('loss')
        if loss is not None:
            step = trainer.global_step
            with open(self.log_path, 'a') as f:
                f.write(f"{step}\t{loss.item():.6f}\n")


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor="info/global_step",
            mode="max",
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'
    loss_log_path = output_dir / "loss_log.txt"
    callbacks.append(LossLoggerCallback(loss_log_path))

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],  # Uncomment for SLURM auto resubmission.
        inference_mode=False if (cfg.mode == "test" and cfg.test.align_pose) else True,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    distiller = None
    if cfg.train.distiller:
        distiller = get_distiller(cfg.train.distiller)
        distiller = distiller.eval()

    # Load the encoder weights.
    if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
        weight_path = cfg.model.encoder.pretrained_weights
        ckpt_weights = torch.load(weight_path, map_location='cpu')
        if 'model' in ckpt_weights:
            ckpt_weights = ckpt_weights['model']
            ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        elif 'state_dict' in ckpt_weights:
            ckpt_weights = ckpt_weights['state_dict']
            ckpt_weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        else:
            raise ValueError(f"Invalid checkpoint format: {weight_path}")

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss),
        step_tracker,
        distiller=distiller,
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()
