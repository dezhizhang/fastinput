import jieba
import torch
import config
from model import FastInputModel

def predict(text):
    """模型预测"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 词表
    with open(config.MODELS_DIR / 'vocab.txt', 'r',encoding='utf-8') as f:
        vocab_list = [line.strip() for line in f.readlines()]

    word2index = {word:index for index, word in enumerate(vocab_list)}
    index2word = {index:word for index, word in enumerate(vocab_list)}




    # 3. 模型
    model = FastInputModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth'))

    # 4. 处理输入
    tokens = jieba.lcut(text)

    indexes = [word2index.get(token,0) for token in tokens]

    input_tensor = torch.tensor([indexes])

    input_tensor = input_tensor.to(device)



    # 预测逻辑
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    top5_indexes = torch.topk(output, 5).indices


    top5_indexes_list = top5_indexes.tolist()

    top5_token = [index2word[index] for index in top5_indexes_list[0]]
    return top5_token


def run_predict():
    while True:
        user_input = input(">")



if __name__ == "__main__":
    top_tokens = predict("我们团队")

