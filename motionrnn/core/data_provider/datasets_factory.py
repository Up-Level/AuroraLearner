from core.data_provider import polar

datasets_map = {
    'polar': polar
}

def data_provider(args, is_training=True):
    if args.dataset_name not in datasets_map:
        raise ValueError(f'Name of dataset unknown {args.dataset_name}')
    train_data_list = args.train_data_paths.split(',')
    valid_data_list = args.valid_data_paths.split(',')

    if args.dataset_name == 'polar':
        test_input_param = {'paths': valid_data_list,
                            'image_width': args.img_width,
                            'minibatch_size': args.batch_size,
                            'seq_length': args.total_length,
                            'input_length': args.input_length,
                            'input_data_type': 'float32',
                            'channel': args.img_channel,
                            'name': args.dataset_name}

        input_handle = datasets_map[args.dataset_name].DataProcess()
        test_input_handle = input_handle.get_test_input_handle(test_input_param)
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_param = {'paths': train_data_list,
                        'image_width': args.img_width,
                        'minibatch_size': args.batch_size,
                        'seq_length': args.total_length,
                        'input_length': args.input_length,
                        'input_data_type': 'float32',
                        'channel': args.img_channel,
                        'name': 'polar'}

            train_input_handle = input_handle.get_train_input_handle(train_input_param)
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle

        return test_input_handle
