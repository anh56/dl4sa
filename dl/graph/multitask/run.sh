python run.py --output_dir=./saved_models/multitask_regcn_dMSRvul_es_mcc_tune_50 \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn ReGCN --epoch 100 --format uni --seed 123456 --early_stopping_metric mcc --max_early_stopping 10 2>&1 | tee -i ./saved_models/multitask_regcn_dMSRvul_es_mcc_tune_50/training_log.txt

python run.py --output_dir=./saved_models/multitask_reggnn_dMSRvul_es_mcc_tune_50 \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn ReGGNN --epoch 100 --format uni --seed 123456 --early_stopping_metric mcc --max_early_stopping 10 2>&1 | tee -i ./saved_models/multitask_reggnn_dMSRvul_es_mcc_tune_50/training_log.txt

python run.py --output_dir=./saved_models/multitask_devign_dMSRvul_es_mcc_tune_50 \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --model devign --epoch 100 --format uni --seed 123456 --early_stopping_metric mcc --max_early_stopping 10 2>&1 | tee -i ./saved_models/multitask_devign_dMSRvul_es_mcc_tune_50/training_log.txt
