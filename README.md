# NTIRE 2019 - Image Enhancement Challenge

## Inference

1 - Change the directory that must be enhanced in the file: "_./json/enhancement\_pwise\_blurred\_n\_10\_lr\_1e-4\_wd\_0.0.json_" at the key "_dataset &rightarrow; params &rightarrow; regen &rightarrow; input_dir_"

2 - Run the code with the command: 

```sh
python Trainer.py --json ./json/enhancement_pwise_blurred_n_10_lr_1e-4_wd_0.0.json --regen
```

3 - Enhanced images will be in the folder: ./checkpoints/enhancement_pwise_blurred_n_10_lr_1e-4_wd_0.0/regen
