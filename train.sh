# python train.py --gpu_id 1 --project erosiveulcer_fine --compound_coef 2 
# python train.py --gpu_id 2 --project erosiveulcer_fine --compound_coef 2 --use_paste_aug False 
python train.py --gpu_id 3 --project erosiveulcer_fine --compound_coef 2 --debug True 
#--resume logs/erosiveulcer_fine/2022-11-22_22-37-11/weight/efficientdet-d2_best.pth.tar