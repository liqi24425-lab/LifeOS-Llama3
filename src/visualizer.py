import matplotlib.pyplot as plt
import json
import seaborn as sns
import os

DATA_FILE = 'data/generated_data_10k.json'

def main():
    if not os.path.exists(DATA_FILE):
        print("Please run data_generator.py first!")
        return

    with open(DATA_FILE, 'r') as f:
        data = json.load(f)

    # 统计分布
    categories = {'Schedule': 0, 'Health': 0}
    for item in data:
        if 'input' in item and 'Health' in item['input']:
            categories['Health'] += 1
        else:
            categories['Schedule'] += 1

    # 画图
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(x=list(categories.keys()), y=list(categories.values()), palette="viridis")
    
    plt.title(f"Distribution of Synthetic Data (N={len(data)})")
    plt.ylabel("Count")
    plt.xlabel("Category")
    
    # 保存图片
    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig('images/data_distribution.png')
    print("✅ Plot saved to images/data_distribution.png")

if __name__ == "__main__":
    main()