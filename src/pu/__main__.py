import argparse
from pu.experiments import run_experiment
from pu.metrics import mknn as run_mknn_comparison

def main():
    parser = argparse.ArgumentParser(description="Platonic Universe Experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for running experiments
    parser_run = subparsers.add_parser("run", help="Run an experiment to generate embeddings.")
    parser_run.add_argument("--model", required=True, help="Model to run inference on (e.g., 'vit', 'dino', 'astropt').")
    parser_run.add_argument("--mode", required=True, help="Dataset to compare to HSC (e.g., 'jwst', 'legacysurvey', 'sdss', 'desi').")
    parser_run.add_argument("--output-dataset", help="Output HuggingFace dataset.")
    parser_run.add_argument("--batch-size", type=int, default=128, help="Batch size for processing.")
    parser_run.add_argument("--num-workers", type=int, default=0, help="Number of data loader workers.")
    parser_run.add_argument("--knn-k", type=int, default=10, help="K value for mutual KNN calculation.")

    # Subparser for running mknn comparisons
    parser_mknn = subparsers.add_parser("compare", help="Run mknn comparison on existing embeddings.")
    parser_mknn.add_argument("parquet_file", help="Path to the Parquet file with embeddings.")

    args = parser.parse_args()

    if args.command == "run":
        run_experiment(args)
    elif args.command == "compare":
        run_mknn_comparison(args.parquet_file)

if __name__ == "__main__":
    main()