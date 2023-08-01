#!/bin/bash
pjsub sub_tests_Icelake.sh

job_completed=0
while [ $job_completed -eq 0 ]; do
    if   [ -f success_OPENMP ]; then S=F
    else S=Q
    fi

    echo '['$S']' 'waiting for job... (checks status every 5s)'

    if [ "$S" == "F" ]; then
        job_completed=1
    else
        sleep 5
    fi
done

echo job completed
