import argparse
import json
from allMethods import run_Clustering


def parse_args():
    parser = argparse.ArgumentParser(description="Run clustering with specified parameters.")

    # Required arguments
    parser.add_argument("method_name",  default="BURST", type=str, help="Clustering method name")
    parser.add_argument("dataset_name", default="UCR_SyntheticControl", type=str, help="Dataset name")

    # Optional arguments
    parser.add_argument("--online", type=lambda x: (str(x).lower() == 'true'), default=True, help="Online mode (True/False)")
    parser.add_argument("--init_size", type=int, default=None, help="Initialization size (default: None)")
    parser.add_argument("--modified_params", type=str, default="{}",
                        help="JSON string representing modified parameters (default: empty dict)")

    args = parser.parse_args()

    # Convert modified_params from JSON string to dictionary
    try:
        print(args.modified_params)
        args.modified_params = json.loads(args.modified_params)
        print(args.modified_params)
    except json.JSONDecodeError:
        print("Error: modified_params should be a valid JSON string.")
        exit(1)

    return args

if __name__ == "__main__":
    args = parse_args()
    res=run_Clustering(args.method_name, args.online, args.dataset_name,
                   modified_params=args.modified_params, init_size=args.init_size)
    print(res)