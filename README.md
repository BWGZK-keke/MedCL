# MedCL: Learning Consistent Anatomy Distribution for Scribble-supervised Medical Image Segmentation
This project is developed for our MedCL 2025 paper: [MedCL: Learning Consistent Anatomy Distribution for Scribble-supervised Medical Image Segmentation]([https://arxiv.org/pdf/2503.22890]). For more information about MedCL, please read the following paper:

<div align=center><img src="CycleMix.png" width="70%"></div>

```
@article{zhang2025medcl,
  title={MedCL: Learning Consistent Anatomy Distribution for Scribble-supervised Medical Image Segmentation},
  author={Zhang, Ke and Patel, Vishal M},
  journal={arXiv preprint arXiv:2503.22890},
  year={2025}
}
```
Please also cite this paper if you are using CycleMix for your research.

# Datasets
1. The MSCMR dataset with mask annotations can be downloaded from [MSCMRseg](https://zmiclab.github.io/zxh/0/mscmrseg19/data.html). scribble annotations of MSCMRseg have been released in [MSCMR_scribbles](https://github.com/BWGZK/CycleMix/tree/main/MSCMR_scribbles). Please cite this paper if you use the scribbles for your research.
2. The scribble-annotated MSCMR dataset used for training could be directly downloaded from [MSCMR_dataset](https://github.com/BWGZK/CycleMix/tree/main/MSCMR_dataset). 
3. The ACDC dataset with mask annotations can be downloaded from [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) and the scribble annotations could be downloaded from [ACDC scribbles](https://vios-s.github.io/multiscale-adversarial-attention-gates/data). Please organize the dataset as the following structure:
```
XXX_dataset/
  -- TestSet/
      --images/
      --labels/
  -- train/
      --images/
      --labels/
  -- val/
      --images/
      --labels/
```

# Usage
1. Set the "dataset" parameter in main.py, to the name of dataset, i.e., "MSCMR_dataset".
2. Set the "output_dir" in main.py, as the path to save the checkpoints. 
3. Download the dataset. Then, Set the dataset path in /data/mscmr.py, line 110, to your data path where the dataset is located in.
4. Check your GPU devices and define CUDA_VISIBLE_DEVICES
5. Start to train by using the following code
```
python main.py 
```

If you have any problems, please feel free to contact us. Thanks for your attention.
