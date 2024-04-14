import pandas as pd
import json
import numpy as np

data_file_path = "../../data/msr_vul.csv"
json_out_file_path = "../input/msr.json"
train_idxs_out_file = "input/train_msr.txt"
test_idxs_out_file = "input/test_msr.txt"
valid_idxs_out_file = "input/valid_msr.txt"


# Access Gained,Attack Origin,Authentication Required,Availability,CVE ID,CVE Page,CWE ID,Complexity,Confidentiality,Integrity,Known Exploits,Publish Date,Score,Summary,Update Date,Vulnerability Classification,add_lines,codeLink,commit_id,commit_message,del_lines,file_name,files_changed,func_after,func_before,lang,lines_after,lines_before,parentID,patch,project,project_after,project_before,vul,vul_func_with_fix

cols = ["project", "lang", "commit_id", "func_before", "vul"]
cols_mapped = ["project", "commit_id", "func", "target"]
msr = pd.read_csv(
    filepath_or_buffer=data_file_path,
    usecols=cols
)
msr.drop(
    msr[msr.lang != "C"].index,
    inplace=True
)
msr.rename(
    columns={
        "func_before": "func",
        "vul": "target"
    },
    inplace=True
)
msr.drop(
    columns=["lang"],
    inplace=True
)
msr.to_json(
    path_or_buf=json_out_file_path,
    # index=False,
    orient="records"
)

from sklearn.model_selection import train_test_split


def split_3(
    df_input: pd.DataFrame,
    stratify_col: str = 'target',
    frac_train: float = 0.8,
    frac_test: float = 0.1,
    frac_val: float = 0.1,
    random_state=None
):
    if frac_train + frac_val + frac_test != 1.0:
        frac_train = 0.8
        frac_test = 0.1
        frac_val = 0.1
        print(f"Invalid ratio, defaulting to train {frac_train}, test {frac_test}, val {frac_val}")

    if stratify_col not in df_input:
        stratify_col = 'target'
        print(f"Invalid col, defaulting to {stratify_col}")

    X = df_input  # Contains all columns.
    y = df_input[[stratify_col]]  # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(
        X,
        y,
        stratify=y,
        test_size=(1.0 - frac_train),
        random_state=random_state
    )

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(
        df_temp,
        y_temp,
        stratify=y_temp,
        test_size=relative_frac_test,
        random_state=random_state
    )

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test

train, val, test = split_3(
    df_input=msr
)
print(train.shape, val.shape, test.shape)

list_train_idxs = list(train.index.unique())
list_test_idxs = list(test.index.unique())
list_val_idxs = list(val.index.unique())

print(len(list_train_idxs), len(list_test_idxs), len(list_val_idxs))

def indexes_to_file(
    df_input: pd.DataFrame,
    out_file: str
):
    idxs = list(df_input.index.unique())
    np.savetxt(out_file, idxs, delimiter="\n", fmt="%s")


indexes_to_file(train, train_idxs_out_file)
indexes_to_file(test, test_idxs_out_file)
indexes_to_file(val, valid_idxs_out_file)
