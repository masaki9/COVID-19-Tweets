import json
import os
import pandas as pd
from pathlib import Path
from shutil import copyfile
from twarc import Twarc


def hydrate(src_dir, dst_dir, temp_dir):
    src_files = os.listdir(src_dir)

    # Create temp csv files in temp_dir containing only tweet ids
    for file_name in src_files:
        if not file_name.endswith(".csv"):
            continue

        file_to_copy = os.path.join(src_dir, file_name)
        copy_file = os.path.join(temp_dir, file_name)

        if not os.path.isfile(copy_file):
            copyfile(file_to_copy, copy_file)

            # Remove second col and keep tweet id col
            data = pd.read_csv(copy_file, usecols=[0])
            data.to_csv(copy_file, index=False)

    twarc = Twarc()  # Ensure you have configured twarc (twarc configure)
    id_file_names = os.listdir(temp_dir)

    for i in id_file_names:
        id_file = os.path.join(temp_dir, i)
        file_name = Path(id_file).stem

        save_file_path = os.path.join(dst_dir, file_name + ".jsonl")

        if not os.path.isfile(save_file_path):
            try:
                for tweet in twarc.hydrate(open(id_file)):
                    save_file = open(save_file_path, "a")
                    json.dump(tweet, save_file)
                    save_file.write("\n")
            except UnicodeDecodeError:
                print("Can't decode data in {}.".format(os.path.basename(save_file_path)))
        else:
            print("{} already exists. Skipping.".format(os.path.basename(save_file_path)))


if __name__ == "__main__":
    Path("data_hydrated/prep/geo-tagged/").mkdir(parents=True, exist_ok=True)
    Path("data_hydrated/prep/regular/").mkdir(parents=True, exist_ok=True)
    Path("data_hydrated/geo-tagged/").mkdir(parents=True, exist_ok=True)
    Path("data_hydrated/regular/").mkdir(parents=True, exist_ok=True)

    src_dir = "data/geo-tagged/"
    dst_dir = "data_hydrated/geo-tagged/"
    temp_dir = "data_hydrated/prep/geo-tagged/"
    hydrate(src_dir, dst_dir, temp_dir)

    # src_dir = "data/regular/"
    # dst_dir = "data_hydrated/regular/"
    # temp_dir = "data_hydrated/prep/regular/"
    # hydrate(src_dir, dst_dir, temp_dir)
