import os.path
import datetime
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from torch.utils.tensorboard import SummaryWriter
from core.utils import metrics
from core.utils import preprocess


def train(model, ims, real_input_flag, configs, itr, writer: SummaryWriter):
    cost = model.train(ims, real_input_flag)
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        cost += model.train(ims_rev, real_input_flag)
        cost = cost / 2
    
    writer.add_scalar("Loss/train", cost, itr)

    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('training loss: ' + str(cost))
        writer.flush()


def test(model, test_input_handle, configs, itr, writer: SummaryWriter):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim = [], []
    csi20, csi30, csi40, csi50 = [], [], [], []
    if configs.img_width == 136:
        ssim_nightside = 0
        ssim_img = 0
        ssim_plot = 0
    iteration_name = itr if isinstance(itr, int) else None

    for i in range(test_input_handle.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        if configs.dataset_name == 'echo' or configs.dataset_name == 'guangzhou':
            csi20.append(0)
            csi30.append(0)
            csi40.append(0)
            csi50.append(0)

    mask_input = configs.input_length

    real_input_flag = np.zeros(
        (configs.batch_size,
         test_input_handle.total_length - mask_input - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    while (test_input_handle.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)

        img_gen = model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = test_input_handle.total_length - configs.input_length
        img_gen_length = img_gen.shape[1]
        img_out = img_gen[:, -output_length:]

        ssim_images = np.zeros_like(img_out)

        # MSE per frame
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            gx = img_out[:, i, :, :, :]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)
            if configs.dataset_name == 'echo' or configs.dataset_name == 'guangzhou':
                csi20[i] += metrics.cal_csi(pred_frm, real_frm, 20)
                csi30[i] += metrics.cal_csi(pred_frm, real_frm, 30)
                csi40[i] += metrics.cal_csi(pred_frm, real_frm, 40)
                csi50[i] += metrics.cal_csi(pred_frm, real_frm, 50)

            for b in range(configs.batch_size):
                score, full = structural_similarity(pred_frm[b], real_frm[b], channel_axis=-1, full=1)
                ssim[i] += score
                ssim_images[b, i] = full

                if configs.img_width == 136:
                    ssim_nightside += structural_similarity(pred_frm[b, :120, 60:120], real_frm[b, :120, 60:120], channel_axis=-1)
                    ssim_img += structural_similarity(pred_frm[b, :120, :120], real_frm[b, :120, :120], channel_axis=-1)
                    ssim_plot += structural_similarity(pred_frm[b, 120:, :120], real_frm[b, 120:, :120], channel_axis=-1)

        # save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in range(test_input_handle.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB) if img_gt.shape[-1] == 1 else img_gt
                cv2.imwrite(file_name, img_gt)

            for i in range(img_gen_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_gen[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                img_pd = cv2.cvtColor(img_pd, cv2.COLOR_BGR2RGB) if img_pd.shape[-1] == 1 else img_pd
                cv2.imwrite(file_name, img_pd)

            for i in range(output_length):
                name = f"ssim{i + 1}.png"
                file_name = os.path.join(path, name)
                img_ssim = ssim_images[0, i, :, :, :]
                img_ssim = np.maximum(img_ssim, 0)
                img_ssim = np.minimum(img_ssim, 1)
                img_ssim = np.uint8(img_ssim * 255)
                img_ssim = cv2.cvtColor(img_ssim, cv2.COLOR_BGR2RGB) if img_ssim.shape[-1] == 1 else img_ssim
                cv2.imwrite(file_name, img_ssim)

            # Join first few known frames with predicted frames
            total_img = np.concatenate([test_ims[:, :configs.input_length], img_gen], axis=1)
            if total_img.shape[-1] == 1:
                # If greyscale convert to RGB
                total_img = total_img.repeat(3, axis=-1)
            # Reorder axes to required format
            video = total_img.transpose([0, 1, 4, 2, 3])
            writer.add_video(f"Prediction/test/{batch_id}", video, iteration_name, fps=0.001)

            for i in range(output_length):
                ssim_comparison = np.concatenate([test_ims[:, configs.input_length + i], img_gen[:, i], ssim_images[:, i]], axis=2)
                if ssim_comparison.shape[-1] == 1:
                    # If greyscale convert to RGB
                    ssim_comparison = ssim_comparison.repeat(3, axis=-1)
                writer.add_images(f"SSIM-Comparison/{batch_id}", ssim_comparison.transpose([0, 3, 1, 2]), i)

        test_input_handle.next()

    avg_mse = avg_mse / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in range(test_input_handle.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(test_input_handle.total_length - configs.input_length):
        print(ssim[i])

    if configs.dataset_name == 'echo' or configs.dataset_name == 'guangzhou':
        csi20 = np.asarray(csi20, dtype=np.float32) / batch_id
        csi30 = np.asarray(csi30, dtype=np.float32) / batch_id
        csi40 = np.asarray(csi40, dtype=np.float32) / batch_id
        csi50 = np.asarray(csi50, dtype=np.float32) / batch_id
        print('csi20 per frame: ' + str(np.mean(csi20)))
        for i in range(test_input_handle.total_length - configs.input_length):
            print(csi20[i])
        print('csi30 per frame: ' + str(np.mean(csi30)))
        for i in range(test_input_handle.total_length - configs.input_length):
            print(csi30[i])
        print('csi40 per frame: ' + str(np.mean(csi40)))
        for i in range(test_input_handle.total_length - configs.input_length):
            print(csi40[i])
        print('csi50 per frame: ' + str(np.mean(csi50)))
        for i in range(test_input_handle.total_length - configs.input_length):
            print(csi50[i])

    num_frames = batch_id * configs.batch_size * (configs.total_length - configs.input_length)

    writer.add_scalar("Loss/test", avg_mse, iteration_name)
    if configs.img_width == 136:
        writer.add_scalar("SSIM/full", np.mean(ssim), iteration_name)
        writer.add_scalar("SSIM/nightside", ssim_nightside / num_frames, iteration_name)
        writer.add_scalar("SSIM/image", ssim_img / num_frames, iteration_name)
        writer.add_scalar("SSIM/plot", ssim_plot / num_frames, iteration_name)
    else:
        writer.add_scalar("SSIM/image", np.mean(ssim), iteration_name)
