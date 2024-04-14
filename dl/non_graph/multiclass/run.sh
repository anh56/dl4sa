python run.py --output_dir=./saved_models/cnn_access_complexity \
    --do_train --do_eval --do_test \
    --train_data_file=../dataset/msr/vul/train_msr_vul.jsonl --eval_data_file=../dataset/msr/vul/valid_msr_vul.jsonl --test_data_file=../dataset/msr/vul/test_msr_vul.jsonl \
    --num_train_epochs 20 --block_size 512 --train_batch_size 1 --eval_batch_size 1 \
    --num_classes 3 --target_class access_complexity --seed 123456  --model cnn 2>&1 | tee train.log
