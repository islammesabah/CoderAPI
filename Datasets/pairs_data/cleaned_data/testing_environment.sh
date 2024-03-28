conda create -y --name testing_env python=3.10 --no-default-packages

conda activate testing_env

pip install pylint datasets pydub googletrans textblob mmcv
pip install langdetect

rm -f results.txt
touch results.txt

for file in Datasets/pairs_data/cleaned_data/code/*;
do
    api=$(echo "$file" | cut -d '/' -f 5)
    echo "................. $api"
    pip install "$api"
    # file_path="$file/*.py"
    for file_path in "$file"/*.py;
    do
        echo "$file_path"
        python Datasets/pairs_data/cleaned_data/add_main.py --file "$file_path"
        pylint --disable=R,C,W "$file_path"_temp.py >> results.txt
        rm "$file_path"_temp.py
    done
    pip uninstall -y "$api"
done

conda deactivate
conda env remove -n testing_env