# AuroraLearner
Code to preprocess auroral images from the Polar spacecraft and magnetic index data, and a modified [MotionRNN](https://arxiv.org/abs/2103.02243) framework to train and test on these images. Partial code is also given to process images taken from the IMAGE spacecraft.

An example of a preprocessed image with a line graph of the SMU index is shown below:

![Polar Combined](images/Polar_combined.png)

## Results
Batches of 32 different image sequences are shown below. The first three frames in this animation are the real frames from the dataset, and the next four are frames predicted by MotionRNN.

Prediction with auroral images only:

![Auroral images only prediction](images/images.gif)

Prediction with the SML index:

![Auroral images only prediction](images/sml.gif)

Prediction with Interplanetary Magnetic Field readings:

![Auroral images only prediction](images/imf.gif)

The SSIM of the auroral image was evaluated at regular intervals during training. Many different permutations of geomagnetic indices and solar wind parameters were used.

![SSIM Graph](images/ssim.png)

The table of results are shown below, ordered descending from highest nightside SSIM. The best result for each metric has been bolded.

| Model                        | Indices       | MSE   | Image SSIM | Plot SSIM   | Nightside SSIM | Overall SSIM  |
|------------------------------|---------------|-------|------------|-------------|----------------|---------------|
| ConvLSTM [Jiang et al. 2023] | None          | N/A   | N/A        | N/A         | **0.812**      | N/A           |
| MotionRNN                    | Solar Wind    | 176   | 0.842      | 0.658       | 0.779          | 0.838         |
| MotionRNN                    | SML & SMU     | 195   | **0.844**  | 0.535       | 0.779          | 0.827         |
| MotionRNN                    | IMF           | 179   | 0.841      | 0.681       | 0.774          | 0.839         |
| MotionRNN                    | None          | **46.8** | 0.840   | N/A         | 0.769          | **0.840**     |
| MotionRNN                    | Bz            | 142   | 0.840      | **0.697**   | 0.767          | 0.840         |
| MotionRNN                    | All           | 263   | 0.837      | 0.657       | 0.763          | 0.837         |
| MotionRNN                    | SML           | 198   | 0.822      | 0.496       | 0.755          | 0.810         |


## Installation and Use
- Python 3.11.2 was used to develop this software, but Python 3.10 may also work.
- PyTorch 2.6.0 should be installed for the correct device.
- Install other requirements using `pip install -r requirements.txt`
- Download Polar dataset images. This project uses the dataset provided by Jiang et al. and can be downloaded from SciDb [here](https://www.scidb.cn/en/detail?dataSetId=6e62c6f43fbc440092441c4e56286dee). This dataset must be placed in the directory `./polar/images_raw`.
- Download geomagnetic indices and solar wind parameter data from [SuperMAG](https://supermag.jhuapl.edu/indices/). The software expects this data as several CSV files placed in `./polar/indices`.
- Generate the datasets to be trained by running the following commands:
    - `python ./polar/process.py indices`
    - `python ./polar/process.py images gs`
    - `python ./polar/process.py combine`
- Train the models by running `./polar-train.ps1` in PowerShell. This file can be modified to tweak various parameters or to change which models get trained.
- Once trained these models are saved in `./checkpoints`. Tests are performed during training but additional tests can be performed by running `./polar-test.ps1`.
- Results are recorded using TensorBoard. This can be started with `tensorboard --logdir runs` and going to the URL that the command provides.