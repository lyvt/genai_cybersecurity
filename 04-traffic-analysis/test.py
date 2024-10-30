import argparse
from uer.utils import *
from uer.utils.config import load_hyperparam
from model import Classifier
import torch
from process_data import count_labels_num, prepare_data
from uer.opts import finetune_opts
import torch.nn as nn
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(labels, predictions):
    """Calculate evaluation metrics for model performance."""
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average='macro')
    recall = metrics.recall_score(labels, predictions, average='macro')
    f1_score = metrics.f1_score(labels, predictions, average='macro')

    # Generate the confusion matrix
    confusion_matrix = metrics.confusion_matrix(labels, predictions)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('confusion_matrix.pdf')
    plt.show()


    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def batch_loader(batch_size, src, tgt, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        yield src_batch, tgt_batch, seg_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size:, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size:]
        seg_batch = seg[instances_num // batch_size * batch_size:, :]
        yield src_batch, tgt_batch, seg_batch


def test():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)
    args = parser.parse_args()
    args.pooling = 'first'
    args.tokenizer = 'bert'
    args.soft_alpha = 0.5
    args.test_path = 'data/cstnet-tls-1.3/fine-tuning/packet/test_dataset.tsv'
    args.train_path = 'data/cstnet-tls-1.3/fine-tuning/packet/train_dataset.tsv'
    args.vocab_path = 'models/encryptd_vocab.txt'
    args.soft_targets = True
    args.batch_size = 1
    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)
    # Build tokenizer.
    args.tokenizer = BertTokenizer(args)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build classification model.
    model = Classifier(args)
    model.load_state_dict(torch.load(args.output_model_path))
    # Load or initialize parameters.
    model = model.to(args.device)

    # Testing phase.
    test_data = prepare_data(args, args.test_path)
    batch_size = args.batch_size
    src = torch.LongTensor([example[0] for example in test_data])
    tgt = torch.LongTensor([example[1] for example in test_data])
    seg = torch.LongTensor([example[2] for example in test_data])
    args.model = model
    model.eval()
    print("Start testing.")
    labels = []
    predictions = []
    for i, (src_batch, tgt_batch, seg_batch) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
        y_pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        y_gt = tgt_batch
        predictions.extend(y_pred.cpu().tolist())
        labels.extend(y_gt.cpu().tolist())

    # Calculate performance metrics
    performance_metrics = calculate_metrics(labels, predictions)

    return performance_metrics


if __name__ == "__main__":
    metrics = test()
    print(metrics)