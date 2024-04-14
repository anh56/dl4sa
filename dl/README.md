This folder contains the implementation for ReGVD (GGNN, GCN) and Devign (GGNN).
Make sure the data is available in the `data` folder.

The models use JSONL format, so additional conversion is needed if csv file is provided.

- Run data splitter and formatter to get the split functions in jsonl format:
    - [data-splitter.py](data-splitter.py): split data into equal parts as json files
    - [preprocess.py](preprocess.py): convert json into jsonl

Or use the data directly from [here](https://drive.google.com/drive/folders/17zrM4V9b8eOuc9-2SC90hF8I72siTpyc?usp=sharing)

To run using sample settings:
```bash
    pip install -r requirements.txt
    cd /graph/multiclass 
    ./run.sh
```

Additional settings and run commands can be found in each folder correspondingly.

This code is partially based on https://github.com/daiquocnguyen/GNN-ReGVD
