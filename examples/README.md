# Some example programs

## Check model segmentation time on one picture
```
python3 check_time.py --model_path ../segmentation/models/edanetlr\=0.001optim\=Adamaxepoch\=38.pth.tar 
```

## Check model segmentation time on directory with pictures

```
python3 check_time_loop.py --model_path ../segmentation/models/edanetlr\=0.001optim\=Adamaxepoch\=38.pth.tar --test_dir ./test-pictures
```

## Start training  
Example for starting training: 
```
python3 train.py --device cuda --lr 4.5e-2 --num_epochs 20 --train_sim_images_r_dir ./pictures/real/ --train_sim_images_m_dir ./pictures/result/ --train_real_images_r_dir ./pictures/JPEGImages/ --train_real_images_m_dir ./pictures/SegmentationClass/ --test_images_r_dir ./pictures/test/real/ --test_images_m_dir ./pictures/test/segment/ --save_model_dir ./EDANet/models/ --save_model_name EDANet --net_name edanet
```

## Starting training in authomatuic mode
```
python3 train-loop.py
```
## Starting segmentation  
The description of the arguments is at the end of the run_segm.py file. Example of running segmentation: 
```
python3 run_segm.py --device cuda --test_dir ./pictures/test/real/ --save_real_dir ./pictures/test_new/real/ --save_segm_dir ./pictures/test_new/result/ --model_path ./EDANetlr=0.045optim=sgdepoch=19.pth.tar --net_name edanet
```

## Checking markup time for multiple pictures
The description of the arguments is at the end of the check_time_loop.py file. Example to run: 
```
python3 check_time_loop.py --device cuda --test_dir ./pictures/test/real/ --model_path ./test_models/edanetlr=0.0007optim=AdamWepoch=33.pth.tar --net_name edanet
```

## Checking markup time for one picture
The description of the arguments is at the end of the check_time_loop.py file. Example to run:  
```
python3 check_time.py --device cuda --model_path ./test_models/dabnetlr=0.0003optim=Adamaxepoch=35.pth.tar --net_name dabnet
```
