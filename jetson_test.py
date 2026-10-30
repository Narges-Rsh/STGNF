import argparse
import torch

from models.STG_NF.model_pose import STG_NF


def main():

    # ============================================================
    # Arguments
    # ============================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained STG-NF checkpoint",
    )

    args = parser.parse_args()

    # ============================================================
    # Device
    # ============================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("========================================")
    print("DEVICE")
    print("========================================")
    print(device)

    # ============================================================
    # STG-NF model parameters
    #
    # ShanghaiTech default configuration
    # Input shape = [B, C, T, V]
    #             = [B, 2, 24, 18]
    # ============================================================

    model_args = {
        "pose_shape": (2, 24, 18),
        "hidden_channels": 0,
        "K": 8,
        "L": 1,
        "R": 3.0,
        "actnorm_scale": 1.0,
        "flow_permutation": "permute",
        "flow_coupling": "affine",
        "LU_decomposed": True,
        "learn_top": False,
        "edge_importance": False,
        "temporal_kernel_size": None,
        "strategy": "uniform",
        "max_hops": 8,
        "device": str(device),
    }

    # ============================================================
    # Create model
    # ============================================================

    model = STG_NF(
        **model_args
    )

    model = model.to(device)

    # ============================================================
    # Print number of parameters
    # ============================================================

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n========================================")
    print("MODEL INFORMATION")
    print("========================================")

    print(
        f"Total parameters     : {total_params:,}"
    )

    print(
        f"Trainable parameters : {trainable_params:,}"
    )

    print(
        f"Total parameters (K) : {total_params / 1000:.2f} K"
    )

    print(
        f"Total parameters (M) : {total_params / 1e6:.4f} M"
    )

    # ============================================================
    # Load checkpoint
    # ============================================================

    print("\n========================================")
    print("LOADING CHECKPOINT")
    print("========================================")

    print(args.checkpoint)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    print(
        "Checkpoint type:",
        type(checkpoint),
    )

    if isinstance(checkpoint, dict):

        print(
            "Checkpoint keys:",
            checkpoint.keys(),
        )

    # ============================================================
    # Extract state dictionary
    # ============================================================

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model" in checkpoint:

            state_dict = checkpoint[
                "model"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # ============================================================
    # Remove "module." prefix if DataParallel was used
    # ============================================================

    clean_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if key.startswith("module."):

            new_key = key[7:]

        clean_state_dict[new_key] = value

    # ============================================================
    # Load pretrained weights
    # ============================================================

    model.load_state_dict(
        clean_state_dict,
        strict=True,
    )
    
    for module in model.modules():

        if module.__class__.__name__ == "ActNorm2d":
            module.initialized = True
            
            
    # ============================================================
    # Mark ActNorm layers as initialized
    # ============================================================

    num_actnorm = 0

    for module in model.modules():

        if "ActNorm" in module.__class__.__name__:

            print(
                "Found ActNorm:",
                module.__class__.__name__
            )

            if hasattr(module, "inited"):

                # Handle both bool and tensor versions
                if torch.is_tensor(module.inited):
                    module.inited.fill_(True)
                else:
                    module.inited = True

                num_actnorm += 1

            elif hasattr(module, "initialized"):

                if torch.is_tensor(module.initialized):
                    module.initialized.fill_(True)
                else:
                    module.initialized = True

                num_actnorm += 1


    print(
        f"ActNorm layers marked initialized: {num_actnorm}"
    )


    model.eval()

    print(
        "\nPretrained STG-NF loaded successfully."
    )

    # ============================================================
    # Create random PyTorch input
    #
    # Shape:
    # [Batch, Channels, Time, Joints]
    #
    # B = 1
    # C = 2     -> x/y coordinates
    # T = 24    -> 24 frames
    # V = 18    -> 18 keypoints
    # ============================================================

    x = torch.randn(
        1,
        2,
        24,
        18,
        dtype=torch.float32,
        device=device,
    )

    print("\n========================================")
    print("INPUT")
    print("========================================")

    print(
        "Input shape:",
        x.shape,
    )

    print(
        "Input dtype:",
        x.dtype,
    )

    print(
        "Input device:",
        x.device,
    )

    print(
        "Input min:",
        x.min().item(),
    )

    print(
        "Input max:",
        x.max().item(),
    )

    # ============================================================
    # Run inference
    # ============================================================

    print("\n========================================")
    print("INFERENCE")
    print("========================================")

    with torch.no_grad():

        z, nll = model(x)

    # ============================================================
    # Output
    # ============================================================

    print("\n========================================")
    print("OUTPUT")
    print("========================================")

    print(
        "Latent output z shape:",
        z.shape,
    )

    print(
        "\nLatent output z:"
    )

    print(z)

    print(
        "\nNLL shape:",
        nll.shape,
    )

    print(
        "NLL:"
    )

    print(nll)

    print(
        "\nNLL value:",
        nll.item(),
    )

    print("\n========================================")
    print("TEST COMPLETED SUCCESSFULLY")
    
    print("========================================")


if __name__ == "__main__":
    main()