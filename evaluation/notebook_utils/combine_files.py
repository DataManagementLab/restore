import glob

import pandas as pd


def combine_files(source_files, target_file):
    all_filenames = [i for i in glob.glob(source_files)]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    combined_csv.to_csv(target_file, index=False, header=True)

    print(f'Combining {len(all_filenames)} files')
