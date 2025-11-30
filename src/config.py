from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

@dataclass
class TrainingConfig:

    # Training
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: Optional[float] = 5.0
    lr_decay_epochs: Optional[int] = 5
    lr_decay_factor: Optional[float] = 0.5

    # Model
    max_frames: int = 16
    chunk_size: int = 4
    lstm_hidden: int = 640
    image_size: int = 256

    # Inputs
    use_side_frames: bool = False
    use_overhead: bool = True
    use_depth: bool = False
    num_workers: int = 2

    # Paths
    data_root: Path = Path("data/nutrition5k_dataset")
    train_csv: Optional[Path] = None
    test_csv: Optional[Path] = None
    resume_from: Optional[Path] = None

    # Multi-task learning
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        'cal': 1.0,
        'mass': 1.0,
        'fat': 1.0,
        'carb': 1.0,
        'protein': 1.0
    })

    # Metrics
    grad_metrics_freq: int = 10
    advanced_metrics_freq: int = 2
    grad_noise_window: int = 50
    activation_sample_rate: int = 3

    log_dir: Optional[Path] = None
    checkpoint_dir: Path = Path("indexes")


    def get_input_channels(self) -> list[str]:
        channels = []

        if self.use_overhead: channels.append("overhead")
        if self.use_depth: channels.append("depth")
        if self.use_side_frames: channels.append("side")

        return channels

def create_config(args, repo_root: Path) -> TrainingConfig:
    config = TrainingConfig(
        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,

        # Model
        max_frames=args.max_frames,
        chunk_size=args.chunk_size,
        lstm_hidden=args.lstm_hidden,
        image_size=args.image_size,

        # Inputs
        use_side_frames=args.use_side_frames,
        use_overhead=args.use_overhead,
        use_depth=args.use_depth,

        # Paths
        data_root=repo_root / "data/nutrition5k_dataset",
        train_csv=Path(args.train_csv) if args.train_csv else repo_root / "indexes/train_official.csv",
        test_csv=Path(args.test_csv) if args.test_csv else repo_root / "indexes/test_official.csv",
        resume_from=Path(args.resume) if args.resume else None,
        checkpoint_dir=repo_root / "checkpoints"
    )

    return config

