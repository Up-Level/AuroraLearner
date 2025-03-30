Set-Variable CUDA_VISIBLE_DEVICES=0
function Test {
    param ($DatasetName)
    python -u motionrnn/run.py `
        --is_training 0 `
        --device cpu `
        --dataset_name polar `
        --train_data_paths ./polar/datasets/gs/"$DatasetName-train.npz" `
        --valid_data_paths ./polar/datasets/gs/"$DatasetName-test.npz" `
        --save_dir checkpoints/polar/"$DatasetName" `
        --gen_frm_dir results/polar/"$DatasetName" `
        --model_name MotionRNN_PredRNN `
        --pretrained_model ./checkpoints/polar-gs/"$DatasetName"/model.ckpt-1000 `
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
        --batch_size 32 `
        --max_iterations 1000 `
        --display_interval 100 `
        --test_interval 100 `
        --snapshot_interval 100
}

Test sml