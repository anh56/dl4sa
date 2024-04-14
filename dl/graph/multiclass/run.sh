python run.py --output_dir=./saved_models/access_vector_reggnn_dMSRvul_es_macro_f1_tune_s10 \
  --do_eval --do_test --do_train \
  --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --num_classes 3 --target_class access_vector 2>&1 | tee -i ./saved_models/access_vector_reggnn_dMSRvul_es_macro_f1_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/access_complexity_reggnn_dMSRvul_es_macro_f1_tune_s10 \
  --do_eval --do_test --do_train \
  --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --num_classes 3 --target_class access_complexity 2>&1 | tee -i ./saved_models/access_complexity_reggnn_dMSRvul_es_macro_f1_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/authentication_reggnn_dMSRvul_es_macro_f1_tune_s10 \
  --do_eval --do_test --do_train \
  --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --num_classes 2 --target_class authentication 2>&1 | tee -i ./saved_models/authentication_reggnn_dMSRvul_es_macro_f1_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/confidentiality_reggnn_dMSRvul_es_macro_f1_tune_s10 \
  --do_eval --do_test --do_train \
  --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --num_classes 3 --target_class confidentiality 2>&1 | tee -i ./saved_models/confidentiality_reggnn_dMSRvul_es_macro_f1_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/integrity_reggnn_dMSRvul_es_macro_f1_tune_s10 \
  --do_eval --do_test --do_train \
  --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --num_classes 3 --target_class integrity 2>&1 | tee -i ./saved_models/integrity_reggnn_dMSRvul_es_macro_f1_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/availability_reggnn_dMSRvul_es_macro_f1_tune_s10 \
  --do_eval --do_test --do_train \
  --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --num_classes 3 --target_class availability 2>&1 | tee -i ./saved_models/availability_reggnn_dMSRvul_es_macro_f1_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/severity_reggnn_dMSRvul_es_macro_f1_tune_s10 \
  --do_eval --do_test --do_train \
  --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --num_classes 3 --target_class severity 2>&1 | tee -i ./saved_models/severity_reggnn_dMSRvul_es_macro_f1_tune_s10_s10/training_log.txt

python run.py --output_dir=./saved_models/access_complexity_reggnn_dMSRvul_es_mcc_tune_s10 \
  --do_eval --do_test --do_train \
  --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class access_complexity 2>&1 | tee -i ./saved_models/access_complexity_reggnn_dMSRvul_es_mcc_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/authentication_reggnn_dMSRvul_es_mcc_tune_s10 \
  --do_eval --do_test --do_train \
  --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --early_stopping_metric mcc --num_classes 2 --target_class authentication 2>&1 | tee -i ./saved_models/authentication_reggnn_dMSRvul_es_mcc_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/confidentiality_reggnn_dMSRvul_es_mcc_tune_s10 \
  --do_eval --do_test --do_train \
  --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class confidentiality 2>&1 | tee -i ./saved_models/confidentiality_reggnn_dMSRvul_es_mcc_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/integrity_reggnn_dMSRvul_es_mcc_tune_s10 \
  --do_eval --do_test --do_train \
  --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class integrity 2>&1 | tee -i ./saved_models/integrity_reggnn_dMSRvul_es_mcc_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/availability_reggnn_dMSRvul_es_mcc_tune_s10 \
  --do_eval --do_test --do_train \
  --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class availability 2>&1 | tee -i ./saved_models/availability_reggnn_dMSRvul_es_mcc_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/severity_reggnn_dMSRvul_es_mcc_tune_s10 \
  --do_eval --do_test --do_train \
  --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class severity 2>&1 | tee -i ./saved_models/severity_reggnn_dMSRvul_es_mcc_tune_s10/training_log.txt



