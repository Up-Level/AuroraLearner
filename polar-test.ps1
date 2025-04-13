param (
    [int]$numTests = 1
)

function Test {
    param ($DatasetName, $Checkpoint)
    python -u motionrnn/run.py `
        --is_training 0 `
        --device cpu `
        --dataset_name polar `
        --train_data_paths ./polar/datasets/gs/"$DatasetName-train.npz" `
        --valid_data_paths ./polar/datasets/gs/"$DatasetName-test.npz" `
        --save_dir checkpoints/polar/test/"$DatasetName" `
        --gen_frm_dir results/polar/test/"$DatasetName" `
        --model_name MotionRNN_PredRNN `
        --pretrained_model ./checkpoints/"$Checkpoint" `
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

echo "Running $numTests tests"
for ($i=0; $i -lt $numTests; $i++) {
    #Test sml polar/sml/model.ckpt-10000
    Test sml polar-gs/sml/model.ckpt-1000
    #Test sml-smu polar-gs/sml-smu/model.ckpt-1000
    #Test bz polar-gs/bz/model.ckpt-1000
    #Test imf polar-gs/imf/model.ckpt-1000
    #Test wind polar-gs/wind/model.ckpt-1000
    #Test all polar-gs/all/model.ckpt-1000
}