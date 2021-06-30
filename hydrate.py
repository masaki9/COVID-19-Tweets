import json
import os
import pandas as pd
from pathlib import Path
from shutil import copyfile
from twarc import Twarc

Path("data_hydrated/prep/geo-tagged/").mkdir(parents=True, exist_ok=True)
Path("data_hydrated/prep/regular/").mkdir(parents=True, exist_ok=True)
Path("data_hydrated/geo-tagged/").mkdir(parents=True, exist_ok=True)
Path("data_hydrated/regular/").mkdir(parents=True, exist_ok=True)

dst_dir = "data_hydrated/prep/geo-tagged/"
dst_dir2 = "data_hydrated/geo-tagged/"
src_dir = "data/geo-tagged/"
src_files = os.listdir(src_dir)

for fname in src_files:
    file_to_copy = os.path.join(src_dir, fname)
    copy_file = os.path.join(dst_dir, fname)

    if not os.path.isfile(copy_file):
        copyfile(file_to_copy, copy_file)

        # Remove second col and keep tweet id col
        data = pd.read_csv(copy_file, usecols=[0])
        data.to_csv(copy_file, index=False)

twarc = Twarc()
id_file_names = os.listdir(dst_dir)

for i in id_file_names:
    id_file = os.path.join(dst_dir, i)
    file_name = Path(id_file).stem

    save_file_path = os.path.join(dst_dir2, file_name + ".jsonl")

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
