# seg_regression

## Singularity

```bash
singularity build --sandbox seg_reg_img singularity_image.def
```

```bash
singularity shell --sandbox --bind /media:/media,$PWD/results:/root/seg_regression/results seg_reg_img
```
