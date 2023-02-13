# semeval23-task10-sexism-detection

To Train the model, you need to run

CUDA_VISIBLE_DEVICES=7 python3 run_train.py --model_name <model cheackpoint>  --learning_rate 1.6e-5 --epoch 4.0 --output_dir <ouput> --data_dir <data path>

To predict and creat submission .csv use:

CUDA_VISIBLE_DEVICES=7 python3 run_predict.py --model_name <model cheackpoint> --output_dir <ouput>--results_dir <directory for the F1, acuracy results>

