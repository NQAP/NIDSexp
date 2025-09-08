python ./src/combine_train_test.py ./origin_dataset/UNSW_NB15_training-set.csv ./origin_dataset/UNSW_NB15_testing-set.csv -o ./extra_dataset/combined.csv --how rows

python ./src/preprocessing.py ./extra_dataset/combined.csv

python ./src/combined_major_and_minor.py ./extra_dataset/minority_class.csv ./extra_dataset/generated_data_1.csv -o ./extra_dataset/balanced_minor_1.csv --how rows

python ./src/combined_major_and_minor.py ./extra_dataset/FCM_1.csv ./extra_dataset/generated_data_2.csv -o ./extra_dataset/combined_2.csv --how rows

conda install -c rapidsai -c nvidia -c conda-forge cudf=23.04 python=3.10 cudatoolkit=12.0