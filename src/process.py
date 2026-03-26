import pandas as pd
import config
import jieba
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def process():
    """数理预处理"""

    # 1. 读取数据文件
    data_dir = config.RAW_DATA_DIR / 'synthesized_.jsonl'
    df = pd.read_json(data_dir, lines=True, orient='records')

    # 2. 提取数据句子
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split('：')[1])

    # 3. 划分数据集
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2)

    # 4. 构建词表
    vocab_set = set()
    for sentence in tqdm(train_sentences, desc="构建词表"):
        vocab_set.update(jieba.lcut(sentence))

    vocab_list = ['<unk>'] + list(vocab_set)

    # 5. 保存词表
    models_dir = config.MODELS_DIR / 'vocab.txt'
    with open(models_dir, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab_list))

    # 6. 构建训练集
    word2index = {word: index for index, word in enumerate(vocab_list)}
    index_train_sentences = [[word2index.get(token, 0) for token in jieba.lcut(sentence)] for sentence in
                             train_sentences]

    train_dataset = []
    for sentence in index_train_sentences:
        for i in range(len(sentence) - config.SEQ_LEN):
            input = sentence[i:i + config.SEQ_LEN]
            target = sentence[i + config.SEQ_LEN]
            train_dataset.append({"input": input, "target": target})


    print(train_dataset[0:3])


if __name__ == '__main__':
    process()
