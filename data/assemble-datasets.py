"""
Assembles all datasets located at the given path. If no
custom path is given, `data/` will be used.

Additionally, a list of desired datasets may be passed in.
If so, only the provided datasets are considered.

Execute:
    python3 data/assemble-datasets.py
"""

import argparse
import csv
import os

parser = argparse.ArgumentParser("Dataset Assembler")

parser.add_argument(
        "-i", "--include-only", nargs="*",
        help="If provided, only these datasets will be used.",
        dest="include_only")

parser.add_argument(
        "-p", "--data-path", default="data/",
        help="Dataset collection path.", dest="data_path")

if __name__ == "__main__":
    args = parser.parse_args()

    include_only = args.include_only
    data_path = args.data_path

    dataset_list = []

    if include_only is not None:
        dataset_list = [
                os.path.join(data_path, dataset) 
                for dataset in include_only]

    else:
        dataset_list = [
                os.path.join(data_path, dataset) 
                for dataset in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, dataset))]

    with open("all-datasets.csv", "w") as all_data_file:
        dataset_writer = csv.writer(all_data_file)

        use_header = True

        for dataset in dataset_list:
            dataset_file = os.path.join(
                    dataset, dataset.split("/")[-1] + ".csv")

            is_header = True

            if os.path.isfile(dataset_file):
                with open(dataset_file, "r") as single_data_file:
                    dataset_reader = csv.reader(single_data_file)

                    for row in dataset_reader:
                        if use_header:
                            row.append("dataset_name")
                            use_header = False
                            is_header = False

                        elif is_header:
                            is_header = False
                            continue

                        else:
                            row.append(dataset.split("/")[-1])

                        dataset_writer.writerow(row)




            

