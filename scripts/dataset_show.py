import json
import re
import tokenize
from datasets import  load_dataset, load_from_disk
import pandas as pd
from transformers import AutoTokenizer


data_path = "datasets/tooluse"
### Data
if "json" in data_path:
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            datasets = [json.loads(line) for line in f if line.strip()]
    except:
        with open(data_path, 'r', encoding='utf-8') as f:
                datasets = json.load(f)
    print(len(datasets))
    print(datasets[0])
    # 打印第一条数据的键
elif "csv" in data_path:
    datasets = pd.read_csv(data_path)
elif "parquet" in data_path:
    datasets = load_dataset("parquet", data_files=data_path)['train']
    print(datasets[0].keys())
    print(len(datasets))
else:
    datasets = load_dataset(data_path)
    print(len(datasets['train']))
    print(datasets['train'][23].keys())


# import os

# local_dir=f"data/hotpotqa_train_10000/"
# datasets = datasets.select(range(10000))
# datasets.to_parquet(os.path.join(local_dir, "train.parquet"))




