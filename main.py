import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataloader import TextData
from model import Transformer, NoamOpt, SimpleLossCompute
import argparse
import time
import gc
torch.cuda.empty_cache()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='TransformerKVRET_fk_KB_SingleLayer_SubjectTracker_KGAttention',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--gpu', action='store_true', help='use GPU', default=False)
    parser.add_argument('--data_path', type=str, default="data/SMD")
    parser.add_argument('-d', '--hidden_size', default=512, type=int)
    parser.add_argument('-e', '--epochs', default=200, type=int)
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-i', '--d_inner', default=2048, type=int)
    parser.add_argument('-n', '--n_layers', default=1, type=int)
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('-r', '--dropout', default=0.1, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--printevery', default=1, type=int)
    parser.add_argument("--kb_attn", type=str2bool, default=True)
    parser.add_argument("--kvl", type=str2bool, default=True)

    return parser.parse_args(args)


def get_model(args):
    assert args.hidden_size % args.heads == 0
    textdata = args.data
    print("Attention is set to : ", args.kb_attn)
    print("KVL is set to : ", args.kvl)
    model = Transformer(args.hidden_size, args.n_layers, args.heads, args.d_inner, textdata.getTargetMaxLength(),
                        textdata.getVocabularySize(),
                        args.batch_size, textdata.word2id['<sos>'], textdata.word2id['<eos>'],
                        textdata.word2id['<pad>'],
                        None, gpu=args.gpu, lr=args.lr, dropout=args.dropout,
                        use_entity_loss=True, entities_property=textdata.entities_property, kb_attn=args.kb_attn,
                        kvl=args.kvl)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if args.gpu:
        model = model.cuda()

    return model


def main(args):

    best_score = 0

    if args.data_path is None:
        print('Using Default Data path - ./data')
        args.data_path = './data/SMD'
        train_file = args.data_path + '/kvret_train_public.json'
        valid_file = args.data_path + '/kvret_dev_public.json'
        test_file = args.data_path + '/kvret_test_public.json'
        if not Path(train_file).exists() and not Path(valid_file).exists() and not Path(test_file).exists():
            print('Data Files not found at the Default location. Exiting program ...')
            raise FileNotFoundError
    else:
        print('Using data path from argument', args.data_path)
        train_file = args.data_path + '/kvret_train_public.json'
        valid_file = args.data_path + '/kvret_dev_public.json'
        test_file = args.data_path + '/kvret_test_public.json'
        if not Path(train_file).exists() and not Path(valid_file).exists() and not Path(test_file).exists():
            print('Data Files not found at the specified location. Exiting program ...')
            raise FileNotFoundError

    args.data = TextData(train_file, valid_file, test_file, args.data_path)

    print('Datasets Loaded.')

    print('Compiling Model.')

    n_epochs = args.epochs
    epoch = 0

    model = get_model(args)
    model_opt = NoamOpt(512, 1, 4000,
                        torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9))

    if os.path.exists(args.data_path + '/transformer_kg_checkpoint'):
        print("Checkpoint Found. Loading progress from checkpoint ...")
        checkpoint = torch.load(args.data_path + '/transformer_kg_checkpoint')
        model.load_state_dict(checkpoint['model_state_dict'])
        model_opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model_opt._step = checkpoint['step']
        epoch = checkpoint['epoch']
        best_score = checkpoint['best_score']
        print("Epoch trained : ",epoch)
        print("Best Score : ", best_score)

    def get_len(train):
        for i, b in enumerate(train):
            pass
        return i

    print('Model Compiled.')
    print('Training begins')
    while epoch < n_epochs:
        epoch += 1
        epoch_loss = 0
        model.loss = 0
        batches = args.data.getBatches(args.batch_size, transpose=False)
        train_len = get_len(batches)
        model.train()
        for current_batch in tqdm(batches, desc='Processing batches'):
            kb_batch = current_batch.kb_inputs

            input_batch = Variable(torch.LongTensor(current_batch.encoderSeqs))
            out_batch = Variable(torch.LongTensor(current_batch.decoderSeqs))
            input_batch_mask = Variable(torch.FloatTensor(current_batch.encoderMaskSeqs))
            out_batch_mask = Variable(torch.FloatTensor(current_batch.decoderMaskSeqs))
            target_kb_mask = Variable(torch.LongTensor(current_batch.targetKbMask))
            # current_kb_hist = Variable(torch.LongTensor(current_batch.triples_hist))

            decoded_words, loss_Vocab = model.train_batch(input_batch, out_batch, input_batch_mask, out_batch_mask,
                                                          SimpleLossCompute(model.generator, model.criterion,
                                                                               optim=model_opt),
                                                          target_kb_mask=target_kb_mask, kb=kb_batch,
                                                          kb_attn=args.kb_attn, kvl=True)#, current_kb_hist=current_kb_hist)
            model.loss += loss_Vocab
            #del decoded_words, loss_Vocab,input_batch,out_batch,input_batch_mask,out_batch_mask,target_kb_mask,kb_batch
            gc.collect()
            torch.cuda.empty_cache()
            print(torch.cuda.memory_summary(device=None, abbreviated=False))

        epoch_loss = model.loss / train_len
        print(epoch, epoch_loss, "epoch-loss")

        model.eval()
        with torch.no_grad():
            moses_multi_bleu_score, eval_loss = \
                model.evaluate_model(args.data, SimpleLossCompute(model.generator, model.criterion, optim=None),
                                     valid=True, test=False, kb_attn=args.kb_attn, kvl=args.kvl)

        print("Model Bleu using moses_multi_bleu_score :", moses_multi_bleu_score)
        print("Validation Loss :", eval_loss)

        if moses_multi_bleu_score > best_score:
            best_score = moses_multi_bleu_score
            print("Saving Model ...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_opt.optimizer.state_dict(),
                'step': model_opt._step,
                'best_score' : best_score
            }, args.data_path + '/transformer_kg_checkpoint')

    print('Model training complete.')
    model.eval()
    print("Loading best model ...")
    checkpoint = torch.load(args.data_path + '/transformer_kg_checkpoint')
    model.load_state_dict(checkpoint['model_state_dict'])
    model_opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model_opt._step = checkpoint['step']

    with torch.no_grad():
        moses_multi_bleu_score, eval_loss,  all_predicted, target_batches, f1_score = \
            model.evaluate_model(args.data, SimpleLossCompute(model.generator, model.criterion, optim=None), valid=False,
                                 test=True, kb_attn=args.kb_attn, kvl=args.kvl)
        print("Test Model Bleu using moses_multi_bleu_score :", moses_multi_bleu_score)
        print("Model Loss on test:", eval_loss)
        print('Saving Model.')
        test_results = '/test_result.csv'
        test_out = pd.DataFrame()
        test_out['original_response'] = target_batches
        test_out['predicted_response'] = all_predicted
        print('Saving the test predictions......')
        print('F1 Score : ', f1_score)
        test_out.to_csv(args.data_path + test_results, index=False)

if __name__ == '__main__':
    main(parse_args())
