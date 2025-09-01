python ./src/combine_train_test.py ./origin_dataset/UNSW_NB15_training-set.csv ./origin_dataset/UNSW_NB15_testing-set.csv -o ./extra_dataset/combined.csv --how rows

python ./src/preprocessing.py ./extra_dataset/combined.csv

python ./src/combined_major_and_minor.py ./extra_dataset/major_after_reduced.csv ./extra_dataset/generated_data.csv -o ./extra_dataset/combined_oversampling.csv --how rows

conda install -c rapidsai -c nvidia -c conda-forge cudf=23.04 python=3.10 cudatoolkit=12.0