python ./src/combine_train_test.py ./origin_dataset/UNSW_NB15_training-set.csv ./origin_dataset/UNSW_NB15_testing-set.csv -o ./extra_dataset/combined.csv --how rows

python ./src/preprocessing.py ./extra_dataset/combined.csv
