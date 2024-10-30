import torch
import torch.nn as nn
from uer.layers.embeddings import WordPosSegEmbedding
from uer.encoders.transformer_encoder import TransformerEncoder


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = WordPosSegEmbedding(args, len(args.tokenizer.vocab))
        self.encoder = TransformerEncoder(args)
        self.labels_num = args.labels_num
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        # Target.
        output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits
