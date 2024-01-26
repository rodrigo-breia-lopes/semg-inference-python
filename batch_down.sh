#! /bin/sh

data_url="https://ninapro.hevs.ch/files/DB1/Preprocessed/"
dataset_path="./datasets/ninapro/db1/"

if [ ! -d $dataset_path ]; then
    echo "Creating directory..."
    mkdir -p $dataset_path
fi

for (( i=1; i<= 27; i++))
do 
    curl "$data_url""s$i.zip" -o "$dataset_path""s$i.zip"
done
