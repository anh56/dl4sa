from ray.job_submission import JobSubmissionClient
from datetime import datetime
ray_client = JobSubmissionClient(
    address="",
)

ray_job_id = ray_client.submit_job(
    entrypoint="ls -ltah && nvidia-smi && . ~/anaconda3/etc/profile.d/conda.sh && conda create --name python_1759284995734 python=3.11.11 -y && conda activate python_1759284995734 && pip install -r requirements.txt && pip show transformers && pip show tokenizers && python run.py --output_dir=multitask_regcn_dMSRvul_es_mcc_tune_50 --do_eval --do_test --do_train --train_data_file=train_msr_vul.jsonl --eval_data_file=valid_msr_vul.jsonl --test_data_file=test_msr_vul.jsonl --train_batch_size 128 --eval_batch_size 128 --gnn ReGCN --epoch 100 --format uni --seed 123456 --early_stopping_metric mcc --max_early_stopping 10 2>&1 | tee -i multitask_regcn_dMSRvul_es_mcc_tune_50_training_log.txt",
    runtime_env={
        "pip": [
            "tensorboard",
            "mlflow",
            "minio",
            # "torch",
            # "transformers",
            # "datasets",
            # "mlflow",
            # "pandas",
            # "transformers[torch]",
            # "accelerate>=0.26.0",
        ],
        "working_dir": ".",
    },
    entrypoint_num_gpus=1,
    submission_id=f"test_from_client_dl_{datetime.now()}",
    # entrypoint_num_cpus=4,
    # entrypoint_memory=12 * 1024 * 1024 * 1024,
)

# address="http://10.33.205.253:8265",

