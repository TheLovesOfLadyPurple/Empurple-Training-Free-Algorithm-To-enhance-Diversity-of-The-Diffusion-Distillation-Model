import argparse

import torch


def parse_shape(shape_text: str) -> tuple[int, int, int]:
    normalized = shape_text.lower().replace("x", " ").replace(",", " ")
    parts = [int(part) for part in normalized.split()]
    if len(parts) != 3:
        raise ValueError("Shape must have exactly three dimensions, for example: 4x64x64")
    return tuple(parts)


def update_running_sum(norms: torch.Tensor, current_sum: float) -> float:
    return current_sum + norms.sum().item()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate lower and upper bounds over per-round mean L1 norms for Gaussian noise."
    )
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of noise samples per round.")
    parser.add_argument("--num_rounds", type=int, default=10000, help="Number of repeated rounds.")
    parser.add_argument(
        "--shape",
        type=str,
        default="4x64x64",
        help="Noise shape written as CxHxW. Default: 4x64x64.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Chunk size used to avoid materializing all samples at once.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used for the L1 norm calculation, for example cpu or cuda.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if args.num_rounds <= 0:
        raise ValueError("num_rounds must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    shape = parse_shape(args.shape)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    round_mean_l1_norms = []
    overall_lower_bound = float("inf")
    overall_upper_bound = float("-inf")

    print(
        f"Sampling {args.num_rounds} rounds, {args.num_samples} Gaussian noises per round, shape={shape}, batch_size={args.batch_size}, device={device}."
    )

    for round_idx in range(args.num_rounds):
        round_l1_sum = 0.0
        samples_remaining = args.num_samples

        while samples_remaining > 0:
            current_batch_size = min(args.batch_size, samples_remaining)
            noise = torch.randn((current_batch_size, *shape), generator=generator, dtype=torch.float32)
            noise = noise.to(device)
            l1_norms = noise.abs().flatten(start_dim=1).sum(dim=1)
            round_l1_sum = update_running_sum(l1_norms, round_l1_sum)
            samples_remaining -= current_batch_size

        round_mean_l1_norm = round_l1_sum / args.num_samples
        round_mean_l1_norms.append(round_mean_l1_norm)
        overall_lower_bound = min(overall_lower_bound, round_mean_l1_norm)
        overall_upper_bound = max(overall_upper_bound, round_mean_l1_norm)

        print(
            f"Round {round_idx + 1:03d}: mean_l1_norm={round_mean_l1_norm:.6f}"
        )

    round_mean_l1_norms_tensor = torch.tensor(round_mean_l1_norms, dtype=torch.float64)

    print()
    print("Summary across all round means:")
    print(f"Lower bound of the mean L1 norm: {overall_lower_bound:.6f}")
    print(f"Upper bound of the mean L1 norm: {overall_upper_bound:.6f}")
    print(f"Average of the round mean L1 norms: {round_mean_l1_norms_tensor.mean().item():.6f}")


if __name__ == "__main__":
    main()