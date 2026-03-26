import pandas as pd
import config



def process():
    """数理预处理"""

    # 文件所在目录
    data_dir = config.RAW_DATA_DIR / 'synthesized_.jsonl'

    # 读取数据文件
    df = pd.read_json(data_dir, lines=True, orient='records')

    print(df.head())


if __name__ == '__main__':
    process()
