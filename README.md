# NTIRE 2019 - Image Enhancement Challenge

## Inference

1 - Change the directory that must be enhanced in the file: "_enh\_pwise\_single\_filt\_7\_wd\_0.0_" at the key "_dataset_ &rightarrow; _params_ &rightarrow; _regen_ &rightarrow; _input_dir_"

2 - Run the code with the command: 

```sh
python Trainer.py --json ./json/enh_pwise_single_filt_7_wd_0.0.json --regen
```

3 - Enhanced images will be in the folder: "_./checkpoints/enh_pwise_single_filt_7_wd_0.0/regen_"
