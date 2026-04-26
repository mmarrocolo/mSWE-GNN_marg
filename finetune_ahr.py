"""
Fine-tune a pre-trained mSWE-GNN model on the Ahr river dataset.

Usage:
    python finetune_ahr.py --config config_finetune_100m_small.yaml
    python finetune_ahr.py --config config_finetune_100m_small.yaml --epochs 100 --lr 1e-4
"""

import argparse
import os
import time
import torch
import wandb
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch_geometric.loader import DataLoader

from utils.load import read_config
from utils.miscellaneous import (
    get_model, fix_dict_in_config,
    get_numerical_times, get_speed_up, SpatialAnalysis,
)
from utils.dataset import (
    create_model_dataset,
    to_temporal_dataset,
    get_temporal_test_dataset_parameters,
)
from utils.visualization import PlotRollout
from training.train import LightningTrainer, DataModule, CurriculumLearning

torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("high")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_finetune_100m_small.yaml")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override max_epochs from config")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning_rate from config")
    p.add_argument("--output", default="results/finetuned_ahr.h5",
                   help="Path to save the fine-tuned model checkpoint")
    p.add_argument("--checkpoint-dir", default="lightning_logs/finetune_ahr",
                   help="Directory for Lightning checkpoints during training")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = read_config(args.config)

    wandb_logger = WandbLogger(log_model=True, config=cfg)
    wandb.init(config=cfg)
    fix_dict_in_config(wandb)
    config = wandb.config

    if args.epochs is not None:
        config.trainer_options["max_epochs"] = args.epochs
    if args.lr is not None:
        config.lr_info["learning_rate"] = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_parameters = config.dataset_parameters
    scalers = config.scalers
    selected_node_features = config.selected_node_features
    selected_edge_features = config.selected_edge_features

    train_dataset, _, test_dataset, scalers = create_model_dataset(
        scalers=scalers,
        device=device,
        **dataset_parameters,
        **selected_node_features,
        **selected_edge_features,
    )

    temporal_dataset_parameters = config.temporal_dataset_parameters
    temporal_test_dataset_parameters = get_temporal_test_dataset_parameters(
        config, temporal_dataset_parameters
    )

    temporal_train_dataset = to_temporal_dataset(train_dataset, **temporal_dataset_parameters)
    # Use training data as validation proxy (intentional overfitting on single simulation)
    temporal_val_dataset = to_temporal_dataset(
        train_dataset, rollout_steps=-1, **temporal_test_dataset_parameters
    )

    print('Number of training simulations:\t', len(train_dataset))
    print('Number of training samples:\t', len(temporal_train_dataset))
    print('Number of node features:\t', temporal_train_dataset[0].x.shape[-1])
    print('Number of rollout steps:\t', temporal_train_dataset[0].y.shape[-1])
    print('Temporal resolution:\t', dataset_parameters['temporal_res'], 'min')

    num_node_features = temporal_train_dataset[0].x.size(-1)
    num_edge_features = temporal_train_dataset[0].edge_attr.size(-1)
    previous_t = temporal_dataset_parameters["previous_t"]
    max_rollout_steps = temporal_dataset_parameters["rollout_steps"]

    model_parameters = dict(config.models)
    model_type = model_parameters.pop("model_type")

    if model_type == "MSGNN":
        model_parameters["num_scales"] = train_dataset[0].mesh.num_meshes

    model = get_model(model_type)(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        previous_t=previous_t,
        device=device,
        **model_parameters,
    ).to(device)

    trainer_options = dict(config.trainer_options)
    lr_info = dict(config["lr_info"])
    type_loss = trainer_options["type_loss"]

    plmodule_kwargs = {
        "model": model,
        "lr_info": lr_info,
        "trainer_options": trainer_options,
        "temporal_test_dataset_parameters": temporal_test_dataset_parameters,
    }

    if "saved_model" in config and os.path.exists(config["saved_model"]):
        print(f"Loading pre-trained weights from: {config['saved_model']}")
        plmodule = LightningTrainer.load_from_checkpoint(
            config["saved_model"], map_location=device, **plmodule_kwargs
        )
    else:
        print("No saved_model found — training from scratch.")
        plmodule = LightningTrainer(**plmodule_kwargs)

    pldatamodule = DataModule(
        temporal_train_dataset,
        temporal_val_dataset,
        batch_size=trainer_options["batch_size"],
    )

    total_parameters = sum(p.numel() for p in model.parameters())
    wandb.log({"total parameters": total_parameters})
    wandb_logger.watch(model, log="all", log_graph=False)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    curriculum_callback = CurriculumLearning(max_rollout_steps, patience=5)
    early_stopping = EarlyStopping(
        "val_loss", mode="min", patience=trainer_options["patience"], min_delta=1e-5
    )

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=trainer_options["max_epochs"],
        gradient_clip_val=1,
        precision="16-mixed" if device.type == "cuda" else "32",
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, curriculum_callback, early_stopping],
    )

    print("Starting fine-tuning...")
    trainer.fit(plmodule, pldatamodule)

    # Load best checkpoint (fall back to last if val_loss was never improved)
    best_ckpt = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    plmodule = LightningTrainer.load_from_checkpoint(
        best_ckpt, map_location=device, **plmodule_kwargs
    )
    model = plmodule.model.to(device)

    trainer.validate(plmodule, pldatamodule)

    # Save fine-tuned model as a full Lightning checkpoint
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    trainer.save_checkpoint(args.output)
    print(f"Fine-tuned model saved to: {args.output}")

    # --- Full evaluation ---
    test_dataset_name = dataset_parameters["test_dataset_name"]
    temporal_res = dataset_parameters["temporal_res"]
    test_size = len(test_dataset)
    maximum_time = test_dataset[0].WD.shape[1]

    numerical_times = get_numerical_times(
        test_dataset_name + "_test",
        test_size,
        temporal_res,
        maximum_time,
        **temporal_test_dataset_parameters,
        overview_file="database/overview.csv",
    )

    temporal_test_dataset = to_temporal_dataset(
        test_dataset, rollout_steps=-1, **temporal_test_dataset_parameters
    )
    test_dataloader = DataLoader(
        temporal_test_dataset, batch_size=len(temporal_test_dataset), shuffle=False
    )

    start_time = time.time()
    predicted_rollout = trainer.predict(plmodule, dataloaders=test_dataloader)
    prediction_times = (time.time() - start_time) / len(temporal_test_dataset)
    predicted_rollout = [item for roll in predicted_rollout for item in roll]

    spatial_analyser = SpatialAnalysis(
        predicted_rollout, prediction_times,
        test_dataset, **temporal_test_dataset_parameters
    )

    rollout_loss = spatial_analyser._get_rollout_loss(type_loss=type_loss)
    model_times = spatial_analyser.prediction_times

    print('test roll loss WD:', rollout_loss.mean(0)[0].item())
    print('test roll loss V:', rollout_loss.mean(0)[1:].mean().item())

    avg_speedup, std_speedup = get_speed_up(numerical_times, model_times)

    print(f'test CSI_005: {spatial_analyser._get_CSI(water_threshold=0.05).nanmean().item()}')
    print(f'test CSI_03: {spatial_analyser._get_CSI(water_threshold=0.3).nanmean().item()}')

    wandb.log({
        "speed-up": avg_speedup,
        "test roll loss WD": rollout_loss.mean(0)[0].item(),
        "test roll loss V": rollout_loss.mean(0)[1:].mean().item(),
        "test CSI_005": spatial_analyser._get_CSI(water_threshold=0.05).nanmean().item(),
        "test CSI_03": spatial_analyser._get_CSI(water_threshold=0.3).nanmean().item(),
    })

    os.makedirs("results", exist_ok=True)
    fig, _ = spatial_analyser.plot_CSI_rollouts(water_thresholds=[0.05, 0.3])
    plt.savefig("results/CSI.png")
    plt.close('all')

    best_id = rollout_loss.mean(1).argmin().item()
    worst_id = rollout_loss.mean(1).argmax().item()

    for id_dataset, name in zip([best_id, worst_id], ["best", "worst"]):
        rollout_plotter = PlotRollout(
            model.to(device), test_dataset[id_dataset].to(device),
            scalers=scalers, type_loss=type_loss, **temporal_test_dataset_parameters
        )
        if model_type == "MSGNN":
            fig = rollout_plotter.explore_rollout(scale=0)
        else:
            fig = rollout_plotter.explore_rollout()
        plt.savefig(f"results/simulation_{name}.png")
        plt.close('all')

    print("Fine-tuning and evaluation finished!")


if __name__ == "__main__":
    main()
