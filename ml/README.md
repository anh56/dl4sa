Make sure the data is available in the `data` folder. 
To run:
- Using Docker Compose:
  - ```bash
    docker compose up -d
    docker compose exec ml bash
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

