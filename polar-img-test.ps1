param (
    $NumChannel
)

function Test {
    param ($DatasetName, $Checkpoint)
    python -u motionrnn/run.py `
        --is_training 0 `
        --device cuda `
        --dataset_name polar `
        --train_data_paths ./polar/datasets/"$DatasetName-train.npz" `
        --valid_data_paths ./polar/datasets/"$DatasetName-test.npz" `
        --save_dir checkpoints/polar/test/"$DatasetName" `
        --gen_frm_dir results/polar/test/"$DatasetName" `
        --model_name MotionRNN_PredRNN `
        --pretrained_model ./checkpoints/"$Checkpoint" `
        --reverse_input 1 `
        --img_height 120 `
        --img_width 120 `
        --img_channel $NumChannel `
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

Test gs/images polar/gs/images/model.ckpt-1000