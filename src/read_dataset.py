import pandas as pd

df = pd.read_csv('./origin_dataset/UNSW_NB15_training-set.csv')
print(df['attack_cat'].value_counts())