#!/bin/bash
rm -f results.txt
touch results.txt

ProgressBar() {
let _progress=(${1}*100/${2}*100)/100
let _done=(${_progress}*4)/10
let _left=40-$_done
_fill=$(printf "%${_done}s")
_empty=$(printf "%${_left}s")
# 1.2.1 Output example:                           
# 1.2.1.1 Progress : [########################################] 100%
printf "\rProgress : [${_fill// /#}${_empty// /-}] ${_progress}%% [${1}/${2}]" 
}

count=1
_end=0
for file in ./code_files/*;
do
    for file_path in "$file"/*.py;
    do
        ((_end++))
    done
done

for file in ./code_files/*;
do
    api=$(echo "$file" | cut -d '/' -f 3)
    pip install -q "$api"
    for file_path in "$file"/*.py;
    do
        ProgressBar ${count} ${_end}
        ((count++))
        /home/mesabah/anaconda3/bin/python add_main.py --file "$file_path"
        pylint --disable=R,C,W "$file_path"_temp.py >> results.txt
        rm "$file_path"_temp.py
        
    done
done

rm -r ./code_files