LOGFILE=loggers/${1}.log

# CUDA_VISIBLE_DEVICES=1 python3 trainDIR_CV_withCL.py > "$LOGFILE" 2>&1 &

python3 trainCrossView_contrastitive.py > "$LOGFILE" 2>&1 &