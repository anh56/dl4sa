python run.py --output_dir=./saved_models/access_vector --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
    --do_train --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
    --num_train_epochs 10 --block_size 512 --train_batch_size 1 --eval_batch_size 1 \
    --num_classes 3 --target_class access_vector --seed 123456  2>&1 | tee train.log

python run.py --output_dir=./saved_models/multitask-cnn \
  --do_train --do_eval --do_test \
  --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
  --num_train_epochs 20 --block_size 512 --train_batch_size 2 --eval_batch_size 2 \
  --model=cnn --seed 123456 2>&1 | tee train_cnn.log

python run.py --output_dir=./saved_models/multitask-lstm-e20-lr2e-5 \
  --do_train --do_eval --do_test \
  --train_data_file=../../input/train_msr_vul.jsonl --eval_data_file=../../input/valid_msr_vul.jsonl --test_data_file=../../input/test_msr_vul.jsonl \
  --num_train_epochs 20 --block_size 512 --train_batch_size 2 --eval_batch_size 2 \
  --model=lstm --seed 123456 2>&1 | tee train_lstm.log
