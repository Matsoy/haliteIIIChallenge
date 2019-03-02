NB_MAPS=20
SEED=1542502263
WIDTHS=("32" "40" "48" "56" "64")

echo "------------------------------------------------------------------------------"
echo "------------------------------------------------------------------------------"

for i in $( seq 1 $NB_MAPS )
do
    RANDOMSEED=`date +%N|sed s/...$//`
    WIDTH=${WIDTHS[$RANDOM % ${#WIDTHS[@]}]}
    echo "Map #$i generated, size: $WIDTH * $WIDTH, seed: $RANDOMSEED, results will be generated below"
    ./halite --replay-directory replays/ -vvv --seed $RANDOMSEED --width $WIDTH --height $WIDTH "python3 MyBot.py" "python3 MyBot.py" 2>&1 >/dev/null | grep 'rank'
    echo ""
done

echo "------------------------------------------------------------------------------"
echo "------------------------------------------------------------------------------"