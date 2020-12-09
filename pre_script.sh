#################################################
# @Author: Mr. Yan
# @Description: use multiple workers to preprocess each dataset 
#               and it will take several hours.
#################################################

if [ $1 = "VD" ]; then
    # Volleyball Dataset (VD)
    # ids: 0~54 
    nohup python STEP_ZERO.py --dataset_name 'VD' --interval 0 10 &
    nohup python STEP_ZERO.py --dataset_name 'VD' --interval 10 20 &
    nohup python STEP_ZERO.py --dataset_name 'VD' --interval 20 30 &
    nohup python STEP_ZERO.py --dataset_name 'VD' --interval 30 40 &
    nohup python STEP_ZERO.py --dataset_name 'VD' --interval 40 50 &
    nohup python STEP_ZERO.py --dataset_name 'VD' --interval 50 55
    # get the train_test files
    python STEP_ZERO.py --dataset_name 'VD'
elif [ $1 = "CAD" ]; then
    # Collective Activity Dataset (CAD)
    # ids: 1~44
    nohup python STEP_ZERO.py --dataset_name 'CAD' --interval (1,10) &
    nohup python STEP_ZERO.py --dataset_name 'CAD' --interval (10,20) &
    nohup python STEP_ZERO.py --dataset_name 'CAD' --interval (20,30) &
    nohup python STEP_ZERO.py --dataset_name 'CAD' --interval (30,45)
    # get the train_test files
    python STEP_ZERO.py --dataset_name 'CAD'
fi