#python run.py --output_dir=./saved_models/access_vector_regcn_dMSRvul_es_macro_f1_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --num_classes 3 --target_class access_vector 2>&1 | tee -i ./saved_models/access_vector_regcn_dMSRvul_es_macro_f1_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/access_complexity_regcn_dMSRvul_es_macro_f1_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --num_classes 3 --target_class access_complexity 2>&1 | tee -i ./saved_models/access_complexity_regcn_dMSRvul_es_macro_f1_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/authentication_regcn_dMSRvul_es_macro_f1_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --num_classes 2 --target_class authentication 2>&1 | tee -i ./saved_models/authentication_regcn_dMSRvul_es_macro_f1_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/confidentiality_regcn_dMSRvul_es_macro_f1_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --num_classes 3 --target_class confidentiality 2>&1 | tee -i ./saved_models/confidentiality_regcn_dMSRvul_es_macro_f1_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/integrity_regcn_dMSRvul_es_macro_f1_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --num_classes 3 --target_class integrity 2>&1 | tee -i ./saved_models/integrity_regcn_dMSRvul_es_macro_f1_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/availability_regcn_dMSRvul_es_macro_f1_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --num_classes 3 --target_class availability 2>&1 | tee -i ./saved_models/availability_regcn_dMSRvul_es_macro_f1_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/severity_regcn_dMSRvul_es_macro_f1_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --num_classes 3 --target_class severity 2>&1 | tee -i ./saved_models/severity_regcn_dMSRvul_es_macro_f1_tune_s10_s10/training_log.txt


##

#python run.py --output_dir=./saved_models/access_vector_regcn_dMSRvul_es_mcc_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class access_vector  2>&1 | tee -i ./saved_models/access_vector_regcn_dMSRvul_es_mcc_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/access_complexity_regcn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class access_complexity 2>&1 | tee -i ./saved_models/access_complexity_regcn_dMSRvul_es_macro_f1/training_log.txt
#
## authentication has only 2 class
#python run.py --output_dir=./saved_models/authentication_regcn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 2 --target_class authentication 2>&1 | tee -i ./saved_models/authentication_regcn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/confidentiality_regcn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class confidentiality 2>&1 | tee -i ./saved_models/confidentiality_regcn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/integrity_regcn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class integrity 2>&1 | tee -i ./saved_models/integrity_regcn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/availability_regcn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class availability 2>&1 | tee -i ./saved_models/availability_regcn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/severity_regcn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class severity 2>&1 | tee -i ./saved_models/severity_regcn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/access_vector_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class access_vector 2>&1 | tee -i ./saved_models/access_vector_reggnn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/access_complexity_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class access_complexity 2>&1 | tee -i ./saved_models/access_complexity_reggnn_dMSRvul_es_macro_f1/training_log.txt
#
## authentication has only 2 class
#python run.py --output_dir=./saved_models/authentication_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 2 --target_class authentication 2>&1 | tee -i ./saved_models/authentication_reggnn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/confidentiality_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class confidentiality 2>&1 | tee -i ./saved_models/confidentiality_reggnn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/integrity_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class integrity 2>&1 | tee -i ./saved_models/integrity_reggnn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/availability_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class availability 2>&1 | tee -i ./saved_models/availability_reggnn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/severity_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class severity 2>&1 | tee -i ./saved_models/severity_reggnn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/access_vector_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class access_vector 2>&1 | tee -i ./saved_models/access_vector_reggnn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/access_complexity_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class access_complexity 2>&1 | tee -i ./saved_models/access_complexity_reggnn_dMSRvul_es_macro_f1/training_log.txt
#
## authentication has only 2 class
#python run.py --output_dir=./saved_models/authentication_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 2 --target_class authentication 2>&1 | tee -i ./saved_models/authentication_reggnn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/confidentiality_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class confidentiality 2>&1 | tee -i ./saved_models/confidentiality_reggnn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/integrity_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class integrity 2>&1 | tee -i ./saved_models/integrity_reggnn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/availability_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class availability 2>&1 | tee -i ./saved_models/availability_reggnn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/severity_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class severity 2>&1 | tee -i ./saved_models/severity_reggnn_dMSRvul_es_macro_f1/training_log.txt

#python run.py --output_dir=./saved_models/access_vector_devign_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --model devign --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class access_vector 2>&1 | tee -i ./saved_models/access_vector_devign_dMSRvul_es_macro_f1/training_log.txt

#python run.py --output_dir=./saved_models/access_complexity_devign_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --model devign --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class access_complexity 2>&1 | tee -i ./saved_models/access_complexity_devign_dMSRvul_es_macro_f1/training_log.txt
#
## authentication has only 2 class
#python run.py --output_dir=./saved_models/authentication_devign_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --model devign --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 2 --target_class authentication 2>&1 | tee -i ./saved_models/authentication_devign_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/confidentiality_devign_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --model devign --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class confidentiality 2>&1 | tee -i ./saved_models/confidentiality_devign_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/integrity_devign_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --model devign --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class integrity 2>&1 | tee -i ./saved_models/integrity_devign_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/availability_devign_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --model devign --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class availability 2>&1 | tee -i ./saved_models/availability_devign_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/severity_devign_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --model devign --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class severity 2>&1 | tee -i ./saved_models/severity_devign_dMSRvul_es_macro_f1/training_log.txt


# rerun authentication
#python run.py --output_dir=./saved_models/authentication_regcn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 2 --target_class authentication 2>&1 | tee -i ./saved_models/authentication_regcn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/authentication_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 2 --target_class authentication 2>&1 | tee -i ./saved_models/authentication_reggnn_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/authentication_reggnn_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 2 --target_class authentication 2>&1 | tee -i ./saved_models/authentication_reggnn_dMSRvul_es_macro_f1/training_log.txt

# rerun missing comp
#python run.py --output_dir=./saved_models/access_vector_devign_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --model devign --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 3 --target_class access_vector 2>&1 | tee -i ./saved_models/access_vector_devign_dMSRvul_es_macro_f1/training_log.txt
#
#python run.py --output_dir=./saved_models/authentication_devign_dMSRvul_es_macro_f1 \
#  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGGNN --model devign --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
#  --seed 123456 --log_neptune --num_classes 2 --target_class authentication 2>&1 | tee -i ./saved_models/authentication_devign_dMSRvul_es_macro_f1/training_log.txt


# tune with different arch

#python run.py --output_dir=./saved_models/access_vector_regcn_dMSRvul_es_macro_f1_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --num_classes 3 --target_class access_vector 2>&1 | tee -i ./saved_models/access_vector_regcn_dMSRvul_es_macro_f1_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/access_complexity_regcn_dMSRvul_es_macro_f1_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --num_classes 3 --target_class access_complexity 2>&1 | tee -i ./saved_models/access_complexity_regcn_dMSRvul_es_macro_f1_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/authentication_regcn_dMSRvul_es_macro_f1_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --num_classes 2 --target_class authentication 2>&1 | tee -i ./saved_models/authentication_regcn_dMSRvul_es_macro_f1_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/confidentiality_regcn_dMSRvul_es_macro_f1_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --num_classes 3 --target_class confidentiality 2>&1 | tee -i ./saved_models/confidentiality_regcn_dMSRvul_es_macro_f1_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/integrity_regcn_dMSRvul_es_macro_f1_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --num_classes 3 --target_class integrity 2>&1 | tee -i ./saved_models/integrity_regcn_dMSRvul_es_macro_f1_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/availability_regcn_dMSRvul_es_macro_f1_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --num_classes 3 --target_class availability 2>&1 | tee -i ./saved_models/availability_regcn_dMSRvul_es_macro_f1_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/severity_regcn_dMSRvul_es_macro_f1_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --num_classes 3 --target_class severity 2>&1 | tee -i ./saved_models/severity_regcn_dMSRvul_es_macro_f1_tune_s10_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/access_complexity_regcn_dMSRvul_es_mcc_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class access_complexity 2>&1 | tee -i ./saved_models/access_complexity_regcn_dMSRvul_es_mcc_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/authentication_regcn_dMSRvul_es_mcc_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --early_stopping_metric mcc --num_classes 2 --target_class authentication 2>&1 | tee -i ./saved_models/authentication_regcn_dMSRvul_es_mcc_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/confidentiality_regcn_dMSRvul_es_mcc_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class confidentiality 2>&1 | tee -i ./saved_models/confidentiality_regcn_dMSRvul_es_mcc_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/integrity_regcn_dMSRvul_es_mcc_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class integrity 2>&1 | tee -i ./saved_models/integrity_regcn_dMSRvul_es_mcc_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/availability_regcn_dMSRvul_es_mcc_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class availability 2>&1 | tee -i ./saved_models/availability_regcn_dMSRvul_es_mcc_tune_s10/training_log.txt
#
#python run.py --output_dir=./saved_models/severity_regcn_dMSRvul_es_mcc_tune_s10 \
#  --model_type=roberta \
#  --do_eval --do_test --do_train \
#  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
#  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
#  --gnn ReGCN --epoch 100 --format uni \
#  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class severity 2>&1 | tee -i ./saved_models/severity_regcn_dMSRvul_es_mcc_tune_s10/training_log.txt

########################################## tune with reggnn
python run.py --output_dir=./saved_models/access_vector_reggnn_dMSRvul_es_macro_f1_tune_s10 \
  --model_type=roberta \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --num_classes 3 --target_class access_vector 2>&1 | tee -i ./saved_models/access_vector_reggnn_dMSRvul_es_macro_f1_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/access_complexity_reggnn_dMSRvul_es_macro_f1_tune_s10 \
  --model_type=roberta \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --num_classes 3 --target_class access_complexity 2>&1 | tee -i ./saved_models/access_complexity_reggnn_dMSRvul_es_macro_f1_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/authentication_reggnn_dMSRvul_es_macro_f1_tune_s10 \
  --model_type=roberta \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --num_classes 2 --target_class authentication 2>&1 | tee -i ./saved_models/authentication_reggnn_dMSRvul_es_macro_f1_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/confidentiality_reggnn_dMSRvul_es_macro_f1_tune_s10 \
  --model_type=roberta \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --num_classes 3 --target_class confidentiality 2>&1 | tee -i ./saved_models/confidentiality_reggnn_dMSRvul_es_macro_f1_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/integrity_reggnn_dMSRvul_es_macro_f1_tune_s10 \
  --model_type=roberta \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --num_classes 3 --target_class integrity 2>&1 | tee -i ./saved_models/integrity_reggnn_dMSRvul_es_macro_f1_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/availability_reggnn_dMSRvul_es_macro_f1_tune_s10 \
  --model_type=roberta \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --num_classes 3 --target_class availability 2>&1 | tee -i ./saved_models/availability_reggnn_dMSRvul_es_macro_f1_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/severity_reggnn_dMSRvul_es_macro_f1_tune_s10 \
  --model_type=roberta \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --num_classes 3 --target_class severity 2>&1 | tee -i ./saved_models/severity_reggnn_dMSRvul_es_macro_f1_tune_s10_s10/training_log.txt

python run.py --output_dir=./saved_models/access_complexity_reggnn_dMSRvul_es_mcc_tune_s10 \
  --model_type=roberta \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class access_complexity 2>&1 | tee -i ./saved_models/access_complexity_reggnn_dMSRvul_es_mcc_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/authentication_reggnn_dMSRvul_es_mcc_tune_s10 \
  --model_type=roberta \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --early_stopping_metric mcc --num_classes 2 --target_class authentication 2>&1 | tee -i ./saved_models/authentication_reggnn_dMSRvul_es_mcc_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/confidentiality_reggnn_dMSRvul_es_mcc_tune_s10 \
  --model_type=roberta \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class confidentiality 2>&1 | tee -i ./saved_models/confidentiality_reggnn_dMSRvul_es_mcc_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/integrity_reggnn_dMSRvul_es_mcc_tune_s10 \
  --model_type=roberta \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class integrity 2>&1 | tee -i ./saved_models/integrity_reggnn_dMSRvul_es_mcc_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/availability_reggnn_dMSRvul_es_mcc_tune_s10 \
  --model_type=roberta \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class availability 2>&1 | tee -i ./saved_models/availability_reggnn_dMSRvul_es_mcc_tune_s10/training_log.txt

python run.py --output_dir=./saved_models/severity_reggnn_dMSRvul_es_mcc_tune_s10 \
  --model_type=roberta \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
  --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn reggnn --epoch 100 --format uni \
  --seed 123456 --early_stopping_metric mcc --num_classes 3 --target_class severity 2>&1 | tee -i ./saved_models/severity_reggnn_dMSRvul_es_mcc_tune_s10/training_log.txt



