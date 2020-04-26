#!/bin/bash

rm configs/*.json
python script/run_config.py

# specify GPU ids you want to use.
GPUS=('0')
# make gpu process ids the same as your number of gpus.
# all gpus will look for the next untrained task.
GPU_PIDS=('-1')


for CONFIG in configs/*.json
do
    while true
    do
        FREE="False"
        for IDX in ${!GPUS[@]}
        do
            GPU=${GPUS[IDX]}
            PID=${GPU_PIDS[IDX]}
            if [ ${PID} == -1 ] ; then
                FREE="True"
                break
            fi
            if [ ${PID} != -1 ] && ! ps -p ${PID} > /dev/null ; then
                echo "waiting gpu" ${GPU} "on pid" ${PID}
                wait ${PID}
                FREE="True"
                break
            fi
        done
        if [ ${FREE} == "True" ] ; then
            break
        else
            sleep 1
            echo "running bash from main process" $$
        fi
    done
    
    bash script/ft.sh ${CONFIG} ${GPU} 10 &
    GPU_PIDS[${IDX}]=$!
done

for IDX in ${!GPUS[@]};
do
    GPU=${GPUS[IDX]}
    PID=${GPU_PIDS[IDX]}
    echo "waiting gpu" ${GPU} "on pid" ${PID}
    wait ${PID}
done

python src/report.py
