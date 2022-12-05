# EfficientDet_Pytorch_Dev
This repo is forked and edited based on https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git

## new features

- add a new dataloader copy and paste augmentation 
```efficientdet/dataset.py/FakeCocoDataset```

- add a new evaluation file
```eval.py```

- add training event visualization, model initialization, etc.
```train.py```

## train

```train.sh```

```
python train.py --gpu_id 2 --project erosiveulcer_fine --compound_coef 2 --use_paste_aug False \
--resume logs/erosiveulcer_fine/2022-12-02_11-56-55/weight/efficientdet-d2.pth.tar
```

### evaluate
```python eval.py```

## others(push a local repo to another remote repo)
- git remote add [remote_name] [remote_branch_name]
- git remote set-url [remote_name] [remote repo url]
- git push [remote_name] [local_branch]:[remote_branch_name]
- git branch --set-upstream-to=[remote_name]/[remote_branch_name] #set default push repo