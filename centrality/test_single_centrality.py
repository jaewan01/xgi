import argparse
import time
import xgi
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for XGI centrality algorithms.")
    parser.add_argument('--dataset', type=str, required=True, help="Specify the dataset to use.")
    parser.add_argument('--measure', type=str, required=True, help="Specify the centrality measure to test.")
    parser.add_argument('--edge', default=False, action='store_true', help="Specify if the test is for edge centrality.")
    args = parser.parse_args()

    hypergraph = xgi.load_xgi_data(args.dataset, read=True, path="datasets")
    print("loaded hypergraph ", args.dataset)

    if not args.edge:
        time_start = time.time()
        if args.measure == "degree":
            centralities = hypergraph.nodes.degree_centrality.asnumpy()
        time_end = time.time()
        print(f"Time taken for node_{args.measure}: {time_end - time_start} seconds")
    else:
        time_start = time.time()
        if args.measure == "degree":
            centralities = hypergraph.edges.degree_centrality.asnumpy()
        time_end = time.time()
        print(f"Time taken for edge_{args.measure}: {time_end - time_start} seconds")

    assert np.sum(centralities) == 1

    if args.edge:
        measure_name = "edge_" + args.measure
    else:
        measure_name = "node_" + args.measure

    with open("runtime.txt", "a+") as f:
        f.write(f"{args.dataset} {measure_name} {time_end - time_start:.4f}s\n")

    os.makedirs(f"values/{args.dataset}/", exist_ok=True)
    np.save(f"values/{args.dataset}/{measure_name}.npy", centralities)