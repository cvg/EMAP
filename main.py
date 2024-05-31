import torch
import argparse
import logging
import numpy as np
import random
from pyhocon import ConfigFactory
from src.runner.runner_udf import Runner_UDF


def fix_random_seeds(seed=42):
    """
    Fix the random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_runner(mode):
    """
    Get the runner based on the provided mode.
    """
    runners = {
        "udf": Runner_UDF,
    }
    if mode not in runners:
        raise ValueError(f"Unknown mode: {mode}")
    return runners[mode]


def main():
    """
    Main function to parse arguments and run the appropriate mode.
    """
    torch.set_default_dtype(torch.float32)
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()

    # Parameters for the training
    parser.add_argument(
        "--conf", type=str, default="./confs/ABC.conf", help="Path to the config file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "extract_edge"],
        help="Mode to run.",
    )
    parser.add_argument(
        "--scan", type=str, default="null", help="The name of a dataset."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use.")
    parser.add_argument(
        "--is_continue",
        default=False,
        action="store_true",
        help="Flag to continue training.",
    )

    args = parser.parse_args()

    # Fix the random seed
    fix_random_seeds()

    with open(args.conf, "r") as f:
        conf_text = f.read()
    conf = ConfigFactory.parse_string(conf_text)

    if args.scan != "null":
        conf["dataset"]["scan"] = args.scan

    logging.info(f"Run on scan of {conf['dataset']['scan']}")

    runner_class = get_runner(conf["general"]["model_type"])
    runner = runner_class(conf, args.mode, args.is_continue, args)

    if args.mode == "train":
        logging.info(f"Training UDF")
        runner.train()
    elif args.mode == "extract_edge":
        logging.info(f"Extracting edges from UDF")
        runner.extract_edge(
            resolution=conf["edge_extraction"]["resolution"],
            udf_threshold=conf["edge_extraction"]["udf_threshold"],
            sampling_N=conf["edge_extraction"]["sampling_N"],
            sampling_delta=conf["edge_extraction"]["sampling_delta"],
            is_pointshift=conf["edge_extraction"]["is_pointshift"],
            iters=conf["edge_extraction"]["iters"],
            is_linedirection=conf["edge_extraction"]["is_linedirection"],
            visible_checking=conf["edge_extraction"]["visible_checking"],
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
