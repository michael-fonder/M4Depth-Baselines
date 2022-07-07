# Baseline for depth estimation on the Mid-Air and TartanAir datasets

For our research, we had to adapt the code of several open-sourced depth estimation methods to train and use them with the Mid-Air and the TartanAir datasets.
We used the results as a baseline for our own depth estimation method, M4Depth.

This repository provides all the source code that was used to generate the baseline which is presented in the following paper:

> **M4Depth: Monocular depth estimation for autonomous vehicles in unseen environments**
>
> [Michaël Fonder](https://www.uliege.be/cms/c_9054334/fr/repertoire?uid=u225873), [Damien Ernst](https://www.uliege.be/cms/c_9054334/fr/repertoire?uid=u030242) and [Marc Van Droogenbroeck](https://www.uliege.be/cms/c_9054334/fr/repertoire?uid=u182591) 
> 
> [arXiv pdf](https://arxiv.org/pdf/2105.09847.pdf)

## Overview

We provide dataloaders for Mid-Air and TartanAir for the following depth estimation methods:
* Monodepth
* Monodepth2
* Manydepth
* ST-CLSTM
* RNN-depth-pose

We also provide a TartanAir dataloader for DeepV2D and an adaptation of PWCDC-Net for depth estimation.

## Instructions

Each directory of this repository is a clone of the original Git repository of each methods.
The requirements are different for each method and should be given in their README file.

To reproduce the baseline presented in our paper, we provide several bash scripts in the `scripts` directory.
With our scripts, you should first train a method on Mid-Air before being able to test it on Mid-Air or TartanAir unstructured.

We provide some utility scripts to download Mid-Air and TartanAir in the `scripts/datasets` directory.
To be able to download the Mid-Air dataset, you will need to get a file listing all the archives to download. For this, the procedure to follow is:
> 1. Go on the [download page of the Mid-Air dataset](https://midair.ulg.ac.be/download.html)
> 2. Select the "Left RGB" and "Stereo Disparity" image types
> 3. Move to the end of the page and enter your email to get the download links (the volume of selected data should be equal to 316.5Go)


Finally, if you want to reproduce the baseline on the TartanAir urban scenes, you have to download the pretrained weights given by the authors of the methods and put them in the `trained_weights/"method name"-kitti` directory.
In case of issue when restoring pretrained weights, you might need to adapt some paths in the bash scripts.

## Disclaimer

The adaptations were made for the sole purpose of creating the baseline mentioned at the beginning of this file.
We did not test our code for other applications. So some further adaptations may be required for other uses.

## Citation

If you use our work in your research, please consider citing the related paper (in addition to the methods and the datasets you use):

```
@article{Fonder2021M4Depth,
  title     = {M4Depth: Monocular depth estimation for autonomous vehicles in unseen environments},
  author    = {Fonder, Michael and Ernst, Damien and Van Droogenbroeck, Marc},
  booktitle = {arXiv},
  month     = {May},
  year      = {2021}
}
```

## License

We license our contribution under the MIT license (see [LICENSE](LICENSE)).

Each method has its own license which shall be respected (see the license file in each method directory).