"""
Fine-tune a pre-trained mSWE-GNN model on the Ahr river dataset.

Usage (local or Colab):
    python finetune_ahr.py
    python finetune_ahr.py --config config_finetune.yaml --epochs 100 --lr 1e-4

The script uses the same train dataset for validation (acceptable when
fine-tuning on a single simulation). Early stopping monitors train loss.
Saves the best checkpoint to results/finetuned_ahr.h5.
"""

import argparse
import os
import torch
import wandb
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch_geometric.loader import DataLoader

from utils.load import read_config
from utils.miscellaneous import get_model, fix_dict_in_config
from utils.dataset import (
    create_model_dataset,
    to_temporal_dataset,
    get_temporal_test_dataset_parameters,
)
from training.train import LightningTrainer, DataModule, CurriculumLearning

torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("high")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_finetune.yaml")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override max_epochs from config")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning_rate from config")
    p.add_argument("--output", default="results/finetuned_ahr.h5",
                   help="Path to save the fine-tuned model")
    p.add_argument("--checkpoint-dir", default="lightning_logs/finetune_ahr",
                   help="Directory for Lightning checkpoints during training")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = read_config(args.config)
    wandb.init(mode="disabled", config=cfg)
    fix_dict_in_config(wandb)
    config = wandb.config

    # CLI overrides
    if args.epochs is not None:
        config.trainer_options["max_epochs"] = args.epochs
    if args.lr is not None:
        config.lr_info["learning_rate"] = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------ dataset
    train_dataset, _, test_dataset, scalers = create_model_dataset(
        scalers=config.scalers,
        device=device,
        **config.dataset_parameters,
        **config.selected_node_features,
        **config.selected_edge_features,
    )

    temporal_dataset_parameters = config.temporal_dataset_parameters
    temporal_test_dataset_parameters = get_temporal_test_dataset_parameters(
        config, temporal_dataset_parameters
    )

    temporal_train_dataset = to_temporal_dataset(
        train_dataset, **temporal_dataset_parameters
    )
    # Use training data as validation proxy (single-simulation fine-tuning)
    temporal_val_dataset = to_temporal_dataset(
        train_dataset, rollout_steps=-1, **temporal_test_dataset_parameters
    )

    print(f"Train samples : {len(temporal_train_dataset)}")
    print(f"Val  samples  : {len(temporal_val_dataset)}")
    print(f"num_node_features: {temporal_train_dataset[0].x.size(-1)}")
    print(f"num_edge_features: {temporal_train_dataset[0].edge_attr.size(-1)}")

    # ------------------------------------------------------------------- model
    num_node_features = temporal_train_dataset[0].x.size(-1)
    num_edge_features = temporal_train_dataset[0].edge_attr.size(-1)

    model_parameters = dict(config.models)
    model_type = model_parameters.pop("model_type")

    if model_type == "MSGNN":
        model_parameters["num_scales"] = train_dataset[0].mesh.num_meshes

    model = get_model(model_type)(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        previous_t=temporal_dataset_parameters["previous_t"],
        device=device,
        **model_parameters,
    ).to(device)

    trainer_options = dict(config.trainer_options)
    lr_info = dict(config["lr_info"])

    plmodule_kwargs = {
        "model": model,
        "lr_info": lr_info,
        "trainer_options": trainer_options,
        "temporal_test_dataset_parameters": temporal_test_dataset_parameters,
    }

    # Load pre-trained weights as starting point
    if "saved_model" in config and os.path.exists(config["saved_model"]):
        print(f"Loading pre-trained weights from: {config['saved_model']}")
        plmodule = LightningTrainer.load_from_checkpoint(
            config["saved_model"], map_location=device, **plmodule_kwargs
        )
    else:
        print("No saved_model found — training from scratch.")
        plmodule = LightningTrainer(**plmodule_kwargs)

    # ----------------------------------------------------------------- trainer
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    curriculum_callback = CurriculumLearning(
        temporal_dataset_parameters["rollout_steps"], patience=5
    )
    early_stopping = EarlyStopping(
        "val_loss",
        mode="min",
        patience=trainer_options["patience"],
        min_delta=1e-5,
    )

    pldatamodule = DataModule(
        temporal_train_dataset,
        temporal_val_dataset,
        batch_size=trainer_options["batch_size"],
    )

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=trainer_options["max_epochs"],
        gradient_clip_val=1,
        precision="16-mixed" if device.type == "cuda" else "32",
        enable_progress_bar=True,
        logger=False,
        callbacks=[checkpoint_callback, curriculum_callback, early_stopping],
    )

    print("Starting fine-tuning...")
    trainer.fit(plmodule, pldatamodule)

    # ------------------------------------------------- load best and save as h5
    best_path = checkpoint_callback.best_model_path
    print(f"Best checkpoint: {best_path}")

    best_plmodule = LightningTrainer.load_from_checkpoint(
        best_path, map_location=device, **plmodule_kwargs
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(best_plmodule.state_dict(), args.output)
    print(f"Fine-tuned model saved to: {args.output}")

    # ------------------------------------------------------ quick eval on test
    temporal_test_dataset = to_temporal_dataset(
        test_dataset, rollout_steps=-1, **temporal_test_dataset_parameters
    )
    test_loader = DataLoader(
        temporal_test_dataset, batch_size=len(temporal_test_dataset), shuffle=False
    )
    results = trainer.validate(best_plmodule, dataloaders=test_loader)
    print("Test metrics:", results)


if __name__ == "__main__":
    main()
