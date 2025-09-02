import numpy as np

# 讀取 .npz 檔案
data = np.load('./UBSW_NB15_Gamo/Results.npz')

# 檢查裡面有哪些 array
print(data.files)  # 會列出所有變數名稱，例如 ['arr_0', 'arr_1']

arrays = {key: data[key] for key in data.files}

for name, arr in arrays.items():
    print(name, arr)