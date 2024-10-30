import random
import argparse
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.opts import finetune_opts
import tqdm
from model import Classifier
import torch
from process_data import count_labels_num, prepare_data
from torch.utils.tensorboard import SummaryWriter


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path,
                                         map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0', 'cuda:3': 'cuda:0'}),
                              strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup, args.train_steps)
    return optimizer, scheduler


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


def train():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)
    args = parser.parse_args()
    args.pooling = 'first'
    args.tokenizer = 'bert'
    args.soft_alpha = 0.5
    args.train_path = 'data/cstnet-tls-1.3/fine-tuning/packet/train_dataset.tsv'
    args.dev_path = 'data/cstnet-tls-1.3/fine-tuning/packet/valid_dataset.tsv'
    args.vocab_path = 'models/encryptd_vocab.txt'
    args.soft_targets = True
    args.epochs_num = 3
    args.batch_size = 1
    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)
    # Build tokenizer.
    args.tokenizer = BertTokenizer(args)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Build classification model.
    model = Classifier(args)
    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)
    model = model.to(args.device)

    # Training phase.
    train_data = prepare_data(args, args.train_path)
    random.shuffle(train_data)
    instances_num = len(train_data)
    batch_size = args.batch_size
    src = torch.LongTensor([example[0] for example in train_data])
    tgt = torch.LongTensor([example[1] for example in train_data])
    seg = torch.LongTensor([example[2] for example in train_data])
    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)
    args.model = model
    print("Start training.")

    writer = SummaryWriter(log_dir='runs')

    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        print('\nEpoch:', epoch)
        model.train()
        running_loss = 0.0
        train_num = 0
        for i, (src_batch, tgt_batch, seg_batch) in enumerate(batch_loader(batch_size, src, tgt, seg)):
            model.zero_grad()
            src_batch = src_batch.to(args.device)
            tgt_batch = tgt_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)
            loss, _ = model(src_batch, tgt_batch, seg_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * src_batch.shape[0]
            train_num += src_batch.shape[0]
        train_loss = running_loss / train_num
        print(f"\tTrain loss: {train_loss:.4f}")
        writer.add_scalar('Loss/Train_epoch', train_loss, epoch)
        torch.save(model.state_dict(), args.output_model_path)
    writer.close()


if __name__ == "__main__":
    train()
