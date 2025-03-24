from core.data_provider import polar

datasets_map = {
    'polar': polar
}

def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  img_width, seq_length, input_length, is_training=True):
    if dataset_name not in datasets_map:
        raise ValueError(f'Name of dataset unknown {dataset_name}')
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')

    if dataset_name == 'polar':
        test_input_param = {'paths': valid_data_list,
                            'image_width': img_width,
                            'minibatch_size': batch_size,
                            'seq_length': seq_length,
                            'input_length': input_length,
                            'input_data_type': 'float32',
                            'channel': 3,
                            'name': 'polar2'}

        input_handle = datasets_map[dataset_name].DataProcess()
        test_input_handle = input_handle.get_test_input_handle(test_input_param)
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_param = {'paths': train_data_list,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'input_length': input_length,
                       'input_data_type': 'float32',
                       'channel': 3,
                       'name': 'polar2'}

            train_input_handle = input_handle.get_train_input_handle(train_input_param)
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle

        return test_input_handle
