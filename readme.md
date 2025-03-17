# 视觉深度学习任务
## 使用
### 数据
将 `origin-data` 文件夹放在 `train.py` 的上一级目录，或直接更改 `train.py` 的数据路径：
``` python
# 定义数据路径
train_data_path = "../origin-data/pneumonia/origin/train_data.npy"
test_data_path = "../origin-data/pneumonia/origin/test_data.npy"
train_label_path = "../origin-data/pneumonia/origin/train_labels.csv"
test_label_path = "../origin-data/pneumonia/origin/test_labels.csv"

```bash
git clone https://github.com/fert-feet/glaucoma-detection.git

# 装好环境后
conda activate env

python train.py
```

