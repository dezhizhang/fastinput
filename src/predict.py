import jieba
import torch
import config
from model import FastInputModel


def predict_batch(model, inputs):
    """
    批量预测
    :param model: 模型
    :param inputs: 输入
    :return: 预测结果
    """
    model.eval()

    with torch.no_grad():
        outputs = model(inputs)

    top5_indexes = torch.topk(outputs, 5).indices
    top5_indexes_list = top5_indexes.tolist()

    return top5_indexes_list


def predict(text, model, word2index, index2word, device):
    """模型预测"""

    # 4. 处理输入
    tokens = jieba.lcut(text)

    indexes = [word2index.get(token, 0) for token in tokens]

    input_tensor = torch.tensor([indexes])

    input_tensor = input_tensor.to(device)

    # 5. 预测逻辑
    top5_indexes_list = predict_batch(model, input_tensor)

    top5_token = [index2word[index] for index in top5_indexes_list[0]]
    return top5_token


def run_predict():
    # 准备资源
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 词表
    with open(config.MODELS_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        vocab_list = [line.strip() for line in f.readlines()]

    word2index = {word: index for index, word in enumerate(vocab_list)}
    index2word = {index: word for index, word in enumerate(vocab_list)}

    # 3. 模型
    model = FastInputModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth'))

    while True:
        user_input = input(">")

        if user_input in ['q', 'quit']:
            print("欢迎下次再来")
            break
        if user_input.strip() == '':
            print("请输入内容")
            continue

        token5 = predict(user_input, model, word2index, index2word, device)

        print(token5)


if __name__ == "__main__":
    run_predict()
