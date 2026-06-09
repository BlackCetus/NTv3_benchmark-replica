import argparse
import yaml
from src.data import prepare_genomics_inputs
from src.utils.logging_utils import get_benchmark_logger


_LOGGER = get_benchmark_logger()

def prefetch(config_path: str = "configs/core_10_tasks.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    tasks = cfg["tasks"]
        
    for task_name, cfg in tasks.items():
        _LOGGER.info(f"Fetching data for {task_name}...")
        prepare_genomics_inputs(
            species=cfg["species"],
            bigwig_file_ids=cfg.get("bigwig_ids"),
            bed_file_ids=cfg.get("bed_ids"),
            data_cache_dir="data" # This is where it will be stored
        )
    _LOGGER.info("All core datasets safely cached!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefetch ntv3 benchmark data.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/core_10_tasks.yaml",
        help="Path to the benchmark YAML config file.",
    )
    args = parser.parse_args()
    prefetch(config_path=args.config)