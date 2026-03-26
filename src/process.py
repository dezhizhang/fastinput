import pandas as pd



def process():
    """数理预处理"""

    # 读取数据文件
    df = pd.read_json('../data/raw/synthesized_.jsonl',lines=True,orient='records')

    print(df.head())





if __name__ == '__main__':
    process()
