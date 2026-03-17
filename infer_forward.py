import argparse
import os
import torch
from torch_geometric.loader import DataLoader

from utils.load import read_config
from utils.dataset import create_model_dataset, to_temporal_dataset
from utils.dataset import get_temporal_test_dataset_parameters
from utils.dataset import use_prediction, apply_boundary_condition
from utils.miscellaneous import get_model
from training.train import LightningTrainer, adapt_batch_training


@torch.no_grad()
def rollout_forward(model, batch, rollout_steps):
    """Run autoregressive rollout without any reference/metric computation."""
    temp = adapt_batch_training(batch)

    dynamic_vars = model.previous_t * model.NUM_WATER_VARS
    assert temp.x.shape[-1] >= dynamic_vars, (
        "The number of dynamic variables is greater than the number of node features"
    )

    predictions = []
    for time_step in range(rollout_steps):
        temp.x[:, -dynamic_vars:] = apply_boundary_condition(
            temp.x[:, -dynamic_vars:],
            temp.BC[:, :, time_step],
            temp.node_BC,
            type_BC=temp.type_BC,
        )
        pred = model(temp)
        temp.x = use_prediction(temp.x, pred, model.previous_t)
        predictions.append(pred)

    return torch.stack(predictions, -1)


def main(config_file, output_dir, max_samples=None, batch_size=1):
    cfg = read_config(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    dataset_parameters = cfg["dataset_parameters"]
    scalers = cfg["scalers"]
    selected_node_features = cfg["selected_node_features"]
    selected_edge_features = cfg["selected_edge_features"]

    # Build datasets and scalers exactly as in the existing pipeline.
    _, _, test_dataset, scalers = create_model_dataset(
        scalers=scalers,
        device=device,
        **dataset_parameters,
        **selected_node_features,
        **selected_edge_features,
    )

    temporal_test_dataset_parameters = get_temporal_test_dataset_parameters(
        cfg, cfg["temporal_dataset_parameters"]
    )
    temporal_test_dataset = to_temporal_dataset(
        test_dataset, rollout_steps=-1, **temporal_test_dataset_parameters
    )

    if max_samples is not None:
        temporal_test_dataset = temporal_test_dataset[:max_samples]

    if len(temporal_test_dataset) == 0:
        raise ValueError("No temporal samples found. Check dataset and temporal settings.")

    # Build model with config-defined architecture.
    model_parameters = dict(cfg["models"])
    model_type = model_parameters.pop("model_type")

    num_node_features = temporal_test_dataset[0].x.size(-1)
    num_edge_features = temporal_test_dataset[0].edge_attr.size(-1)
    previous_t = temporal_test_dataset_parameters["previous_t"]

    if model_type == "MSGNN":
        num_scales = test_dataset[0].mesh.num_meshes
        model_parameters["num_scales"] = num_scales

    model = get_model(model_type)(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        previous_t=previous_t,
        device=device,
        **model_parameters,
    ).to(device)

    trainer_options = cfg["trainer_options"]
    lr_info = cfg["lr_info"]
    plmodule = LightningTrainer(model, lr_info, trainer_options, temporal_test_dataset_parameters)

    if "saved_model" not in cfg:
        raise KeyError("The config must include 'saved_model' for forward inference.")

    plmodule_kwargs = {
        "model": model,
        "lr_info": lr_info,
        "trainer_options": trainer_options,
        "temporal_test_dataset_parameters": temporal_test_dataset_parameters,
    }
    plmodule = plmodule.load_from_checkpoint(cfg["saved_model"], map_location=device, **plmodule_kwargs)
    model = plmodule.model.to(device)
    model.eval()

    dataloader = DataLoader(temporal_test_dataset, batch_size=batch_size, shuffle=False)
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []
    sample_offset = 0

    for batch in dataloader:
        batch = batch.to(device)
        # Batch already contains BC horizon from to_temporal_dataset(..., rollout_steps=-1)
        rollout_steps = batch.y.shape[-1]
        pred_roll = rollout_forward(model, batch, rollout_steps=rollout_steps)

        # Save each graph sample separately for easier downstream use.
        for i in range(batch.num_graphs):
            start = batch.ptr[i]
            end = batch.ptr[i + 1]
            sample_pred = pred_roll[start:end].detach().cpu()
            out_path = os.path.join(output_dir, f"prediction_{sample_offset + i:04d}.pt")
            torch.save(sample_pred, out_path)
            saved_paths.append(out_path)

        sample_offset += batch.num_graphs

    print(f"Saved {len(saved_paths)} rollout prediction file(s) to: {output_dir}")
    if len(saved_paths) > 0:
        print(f"Example output shape: {torch.load(saved_paths[0]).shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forward-only autoregressive inference (no loss/CSI/evaluation)."
    )
    parser.add_argument(
        "--config",
        default="config_finetune.yaml",
        help="Path to YAML config with dataset/model/checkpoint settings.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/forward_predictions",
        help="Directory where .pt rollout predictions will be saved.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of temporal samples to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for forward inference.",
    )
    args = parser.parse_args()

    main(
        config_file=args.config,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )
