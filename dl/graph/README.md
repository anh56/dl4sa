This folder contains the implementation for ReGVD (GGNN, GCN) and Devign (GGNN).
Make sure the data is available in the `data` folder.

The graph model use JSONL format, so additional conversion is needed.

- Run data splitter and formatter to get the split functions in jsonl format:
    - [data-splitter.py](data-splitter.py): split data into equal parts as json files
    - [preprocess.py](preprocess.py): convert json into jsonl

To run:

- Using Docker Compose:
    - ```bash
    docker compose up -d
    docker compose exec dl bash
    ./run.sh
    ```

- Using Docker:
    - ```bash
    docker build -t ml .
    docker run -it ml bash
    ./run.sh
    ```

- Locally:
    - ```bash
    pip install -r requirements.txt
    ./run.sh
    ```
