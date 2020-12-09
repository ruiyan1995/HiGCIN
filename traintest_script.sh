# output the log to the path you want
pth="ckpt/VD/activity/R"
if [ ! -d $pth ]; then
    mkdir -p $pth
fi
python STEP_ONE.py --backbone_name 'resNet18' --save_path $pth > $pth/log

