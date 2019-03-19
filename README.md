# NTIRE2019

## Inference

1 - Change the directory that must be enhanced in the file: "./json/enhancement\_pwise\_blurred\_n\_10\_lr\_1e-4\_wd\_0.0.json" at the key "dataset -> params -> regen -> input_dir"

2 - Run the code with the command: 

```python
python Trainer.py --json ./json/enhancement\_pwise\_blurred\_n\_10\_lr\_1e-4\_wd\_0.0.json --regen
```

3 - Run further refinement
