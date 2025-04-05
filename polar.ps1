Set-Variable CUDA_VISIBLE_DEVICES=0
function Train {
    param ($DatasetName)
    python -u motionrnn/run.py `
        --is_training 1 `
        --device cuda `
        --dataset_name polar `
        --train_data_paths ./polar/gs/datasets/"$DatasetName-train.npz" `
        --valid_data_paths ./polar/gs/datasets/"$DatasetName-test.npz" `
        --save_dir checkpoints/polar/"$DatasetName" `
        --gen_frm_dir results/polar/"$DatasetName" `
        --model_name MotionRNN_PredRNN `
        --reverse_input 1 `
        --img_height 136 `
        --img_width 136 `
        --img_channel 1 `
        --input_length 3 `
        --total_length 5 `
        --num_hidden 64,64,64,64 `
        --filter_size 5 `
        --stride 1 `
        --patch_size 4 `
        --layer_norm 0 `
        --scheduled_sampling 1 `
        --sampling_stop_iter 50000 `
        --sampling_start_value 1.0 `
        --sampling_changing_rate 0.00002 `
        --lr 0.0003 `
        --batch_size 96 `
        --max_iterations 10000 `
        --display_interval 100 `
        --test_interval 100 `
        --snapshot_interval 100
}

Train sml
Train sml-smu
Train bz
Train imf
Train wind
Train all