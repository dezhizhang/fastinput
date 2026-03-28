from torch import nn
import config


class FastInputModel(nn.Module):
    """构建训练模型"""

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM)

        self.rnn = nn.RNN(input_size=config.EMBEDDING_DIM,hidden_size=config.HIDDEN_SIZE,batch_first=True)


        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE,out_features=vocab_size)


    def forward(self,x):
        """前项传播"""
        embed = self.embedding(x)

        output,_ = self.rnn(embed)

        last_hidden_state = output[:,-1,:]

        output = self.linear(last_hidden_state)






