from datetime import datetime
import wandb
import hydra
from omegaconf import DictConfig, open_dict
from .dataset import dataset_factory
from .models import model_factory
from .components import lr_scheduler_factory, optimizers_factory, logger_factory
from .training import training_factory
from datetime import datetime


def model_training(cfg: DictConfig, group_name: str):

    with open_dict(cfg):
        cfg.unique_id = f"{group_name}_{datetime.now().strftime('%m-%d-%H-%M')}"

    dataloaders = dataset_factory(cfg)
    logger = logger_factory(cfg)
    model = model_factory(cfg)
    optimizers = optimizers_factory(model, cfg)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer,
                                         cfg=cfg)
    training = training_factory(cfg, model, optimizers,
                                lr_schedulers, dataloaders, logger)

    training.train()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    layer_num = len(cfg.model.sizes)


    group_name = f"{cfg.dataset.name}_{cfg.model.name}_L{layer_num}_{cfg.datasz.percentage}_{cfg.preprocess.name}"

    if "views" in cfg.dataset:
        view_name = "_".join([view.name for view in cfg.dataset.views])
        group_name = f"{group_name}_{view_name}"

    if "tasks" in cfg.dataset:
        task_name = "_".join([task.name for task in cfg.dataset.tasks])
        task_num = len(cfg.dataset.tasks)
        if len(task_name) > 50:
            task_name = f"MTN_{task_num}"
        group_name = f"{group_name}_{task_name}"

    group_name = group_name if "column" not in cfg.dataset else f"{group_name}_{cfg.dataset.column}"

    group_name = group_name if "ts_length" not in cfg.dataset else f"{group_name}_{cfg.dataset.ts_length}"

    group_name = group_name if cfg.name_appendix is None else f"{group_name}_{cfg.name_appendix}"

    group_name = group_name if "alpha" not in cfg.preprocess else f"{group_name}_{cfg.preprocess.alpha}"

    group_name = f"{group_name}_mask" if "mask" in cfg.model and cfg.model.mask else group_name

    if "use_balance_weight" in cfg.dataset:
        group_name = group_name if not cfg.dataset.use_balance_weight else f"{group_name}_balance_weight"



    if cfg.model.sparse_loss:
        group_name = f"{group_name}_sp_{cfg.model.sparse_loss_weight}"

    if len(group_name) > 90:
        
        group_name = f"{group_name[:90]}..."

    # _{cfg.training.name}\
    # _{cfg.optimizer[0].lr_scheduler.mode}"

    for _ in range(cfg.repeat_time):
        run = wandb.init(project=cfg.project, entity=cfg.wandb_entity, reinit=True,
                         group=f"{group_name}", tags=[f"{cfg.dataset.name}"])

        wandb.config.update({
            "group_name": group_name,
            "dataset": cfg.dataset.name,
            "model": cfg.model.name,
            "datasz": cfg.datasz.percentage,
            "preprocess": cfg.preprocess.name,
            "column": cfg.dataset.column if "column" in cfg.dataset else None,
            "ts_length": cfg.dataset.ts_length if "ts_length" in cfg.dataset else None,
            "alpha": cfg.preprocess.alpha if "alpha" in cfg.preprocess else None,
        })
        model_training(cfg, group_name)

        run.finish()


if __name__ == '__main__':
    main()
