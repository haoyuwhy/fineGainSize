#!/bin/bash
#BSUB -J finegain-res50
#BSUB -q normal
#BSUB -n 1
#BSUB -R span[ptile=1]
#BSUB -o /share/home/MZ2109122/finegain/fineGainSize/logs/log.%J.txt
#BSUB -e /share/home/MZ2109122/finegain/fineGainSize/logs/error.%J.txt
#BSUB -gpu  "num=1:mode=exclusive_process" 
#BSUB -m gpu05
nvidia-smi



python /share/home/MZ2109122/finegain/fineGainSize/main.py --cfg /share/home/MZ2109122/finegain/fineGainSize/configs/resnet50Pre.yaml