#!/bin/bash

set -e
set -x

source /home/jcarter/sandbox/trading/strats/env.sh

utils="/home/jcarter/sandbox/trading/utils"

cmd="python /home/jcarter/sandbox/trading/strats/mp/mp.py"
test_output_dir="/home/jcarter/sandbox/trading/tests/mp/"
test_name="mp_stats.csv"

names=`python $utils/data_catalog.py --file="/home/jcarter/sandbox/trading/data/top100_vol.txt"  --count_limit=2200 --names_only`
for x in $names
do
    $cmd $x | grep "=" >> $test_output_dir/$test_name
done

