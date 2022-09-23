import math
import os
import torch
import torch.nn as nn
import copy
from torch.autograd import Variable
import re
import torch.nn.functional as F
import subprocess
import tempfile
import numpy as np
import gc
from six.moves import urllib
from torch.utils.data import dataloader


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if opt.gpu:
        np_mask = np_mask.cuda()
    return np_mask


def create_masks(src, trg, opt):
    src_mask = (src != opt.pad_tok).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.pad_tok).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size, opt)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x, decoder_self_attn = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x), decoder_self_attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))

        decoder_self_attn = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(decoder_self_attn, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x, decoder_self_attn


class KGAttention(nn.Module):
    def __init__(self, vocab_size, d_model, heads, dropout=0.1,textData=None,gpu=False):
        super().__init__()
        self.data = textData
        self.embed = Embedder(vocab_size, d_model)
        self.gpu=gpu
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.norm = Norm(d_model)

    def forward(self, decoder_self_attn, kg_input, kg_mask):
        # print(kg_input)
        if self.gpu:
            subject = torch.from_numpy(np.array(kg_input)[:, :, 0]).cuda()
            relation = torch.from_numpy(np.array(kg_input)[:, :, 1]).cuda()
            predicate = torch.from_numpy(np.array(kg_input)[:, :, 2]).cuda()
        else:
            subject = torch.from_numpy(np.array(kg_input)[:, :, 0])
            relation = torch.from_numpy(np.array(kg_input)[:, :, 1])
            predicate = torch.from_numpy(np.array(kg_input)[:, :, 2])

        subject_embed = self.embed(subject)
        # sub=self.data.sequence2str(subject[0],tensor=True)
        # pred=self.data.sequence2str(relation[0],tensor=True)
        # obj=self.data.sequence2str(predicate[0],tensor=True)
        relation_emb = self.embed(relation)
        kg_key = subject_embed + relation_emb
        kg_value = self.embed(predicate)
        if self.gpu:
            kg_mask = kg_mask.cuda()
        x = self.attn(decoder_self_attn, kg_key, kg_value, kg_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    def __init__(self, generator, criterion, optim=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = optim

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm

        if self.opt is not None:
            self.opt.optimizer.zero_grad()
            loss.backward()
            self.opt.step()

        return loss.item() * norm


def moses_multi_bleu(hypotheses, references, lowercase=False):
    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    # Get MOSES multi-bleu script
    try:
        multi_bleu_path, _ = urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
            "master/scripts/generic/multi-bleu.perl")
        os.chmod(multi_bleu_path, 0o755)
    except:  # pylint: disable=W0702
        print("Unable to fetch multi-bleu.perl script, using local.")
        metrics_dir = os.path.dirname(os.path.realpath(__file__))
        bin_dir = os.path.abspath(os.path.join(metrics_dir, "..", "..", "bin"))
        multi_bleu_path = os.path.join(bin_dir, "tools/multi-bleu.perl")

    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print("multi-bleu.perl script returned non-zero exit code")
                print(error.output)
                bleu_score = np.float32(0.0)

    hypothesis_file.close()
    reference_file.close()
    return bleu_score


def compute_prf(gold, pred, global_entity_list, kb_plain):
    local_kb_word = [k[0] for k in kb_plain]
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in set(pred):
            if p in local_kb_word or p in global_entity_list:
                if p not in gold:
                    FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    else:
        precision, recall, F1, count = 0, 0, 0, 0
    return F1, count


def compute_f1(gold, pred, global_entity, kb):
    epsilon = 0.000000001
    f1_score = 0.0
    microF1_TRUE = 0.0
    microF1_PRED = 0.0

    for it in range(len(gold)):
        f1_true, count = compute_prf(gold[it], pred[it], global_entity, kb[it])
        microF1_TRUE += f1_true
        microF1_PRED += count

    f1_score = microF1_TRUE / float(microF1_PRED + epsilon)
    return f1_score


class Transformer(nn.Module):
    def __init__(self, hidden_size, n_layers, n_heads, d_inner, max_r, n_words, b_size, sos_tok, eos_tok, pad_tok, itos,textData=None,
                 gpu=False, lr=0.01,
                 dropout=0.1, use_entity_loss=False, entities_property=None, kb_attn=True, kvl=True,double_gen=True):
        super(Transformer, self).__init__()
        self.name = "Transformer"
        self.textData=textData
        self.input_size = n_words
        self.output_size = n_words
        self.hidden_size = hidden_size
        self.d_inner = d_inner
        self.max_r = max_r
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.b_size = b_size
        self.sos_tok = sos_tok
        self.eos_tok = eos_tok
        self.pad_tok = pad_tok
        self.itos = itos
        self.gpu = gpu
        self.lr = lr
        self.use_entity_loss = use_entity_loss
        self.entities_p = entities_property
        self.kb_attn = kb_attn
        self.double_gen = double_gen
        self.kvl = kvl

        self.encoder = Encoder(self.input_size, self.hidden_size, self.n_layers, self.n_heads, self.dropout)
        self.decoder = Decoder(self.output_size, self.hidden_size, self.n_layers, self.n_heads, self.dropout)
        if kb_attn:
            self.kg_attention = KGAttention(self.output_size, self.hidden_size, self.n_heads, self.dropout,gpu=self.gpu)
            self.kg_attention2 = KGAttention(self.output_size, self.hidden_size, self.n_heads, self.dropout, gpu=self.gpu)
        self.generator = Generator(self.hidden_size, self.output_size)
        if double_gen:
            self.generator2 = Generator(self.hidden_size, self.output_size)
        self.criterion = LabelSmoothing(size=self.output_size, padding_idx=self.pad_tok, smoothing=0.1)

        self.loss = 0
        self.print_every = 1

    def train_batch(self,  input_batch, out_batch, input_mask, out_mask, loss_compute, target_kb_mask=None, kb=None,
                    kb_attn=True, kvl=True, current_kb_hist=None):
        batch_loss = 0
        if self.gpu:
            src = input_batch.cuda()
            trg = out_batch.cuda()
        else:
            src = input_batch
            trg = out_batch

        trg_input = trg[:, :-1]
        trg_y = trg[:, 1:]

        max_target_length = trg_input.size(1)

        b_size = trg_input.size(0)

        src_mask, trg_mask = create_masks(src, trg_input, self)

        n_tokens = (trg_y != self.pad_tok).data.sum().item()
        #x=self.textData.sequence2str(src[0],tensor=True)

        encoder_op = self.encoder(src, src_mask)

        target_kb_mask = target_kb_mask.unsqueeze(-2)
        if self.gpu:
            decoded_words = torch.zeros(b_size, int(max_target_length)).cuda()
            decoder_input = torch.zeros(b_size, int(max_target_length)).cuda().long()
            decoder_input[:, 0] = torch.LongTensor([self.sos_tok])
            all_decoder_outputs_vocab = Variable(torch.zeros(b_size, max_target_length, 512)).cuda()
        else:
            decoded_words = torch.zeros(b_size, int(max_target_length))
            decoder_input = torch.zeros(b_size, int(max_target_length)).long()
            decoder_input[:, 0] = torch.LongTensor([self.sos_tok])
            all_decoder_outputs_vocab = Variable(torch.zeros(b_size, max_target_length, 512))

        if kvl:
            subject_tracker = [[] for i in range(self.b_size)]
        if kb_attn and self.double_gen:
            kg_attn2 = self.kg_attention2(encoder_op, kb, target_kb_mask)
            preds2 = self.generator2(kg_attn2)
            # list_v ,list_i = preds2.data.topk(20)
        if kb_attn and not self.double_gen:
            kg_attn2 = self.kg_attention2(encoder_op, kb, target_kb_mask)
            preds2 = self.generator(kg_attn2)

        for j in range(0, max_target_length):
            t_mask = trg_mask[:, :j + 1, :j + 1]

            if j == 0:
                decoder_input_zero = trg_input[:, 0:j + 1]
                decoder_op, decoder_self_attn = self.decoder(decoder_input_zero, encoder_op, src_mask, t_mask)
            else:
                decoder_op, decoder_self_attn = self.decoder(decoder_input[:, 0:j + 1], encoder_op, src_mask, t_mask)

            if kb_attn:
                kg_attn = self.kg_attention(decoder_self_attn, kb, target_kb_mask)
                decoder_op = decoder_op + kg_attn

            preds = self.generator(decoder_op)
            all_decoder_outputs_vocab[:, j] = decoder_op[:, -1, :]
            if kb_attn:
                decoder_vocab = preds[:, -1, :] + preds2[:, -1, :]
            else:
                decoder_vocab = preds[:, -1, :]
            topv, topi = decoder_vocab.data.topk(1)

            if kvl:
                for k in range(self.b_size):
                    topi[k] = self.check_entity(topi[k].item(), kb[k], subject_tracker, k)

            if j != max_target_length - 1:
                if self.gpu:
                    bione = torch.zeros(b_size, int(max_target_length)).cuda().long()
                else:
                    bione = torch.zeros(b_size, int(max_target_length)).long()
                bione[:, j + 1] = Variable(topi.view(-1))
                decoder_input = decoder_input + bione
            decoded_words[:, j] = (topi.view(-1))

        loss = loss_compute(all_decoder_outputs_vocab, trg_y, n_tokens)

        batch_loss = loss / n_tokens
        del loss, src, trg, trg_input, trg_y, max_target_length, decoder_input, all_decoder_outputs_vocab, topi, topv, decoder_input_zero,decoder_self_attn, kb
        del target_kb_mask, preds2, preds,decoder_vocab,decoder_op, src_mask, trg_mask, encoder_op, n_tokens, subject_tracker,t_mask,bione, input_mask, input_batch
        del kg_attn2, kg_attn
        gc.collect()
        torch.cuda.empty_cache()


        return decoded_words, batch_loss

    def evaluate_batch(self, input_batch, out_batch, input_mask, out_mask, loss_compute, target_kb_mask=None, kb=None,
                       kb_attn=True, kvl=True):
        batch_loss = 0
        src = input_batch.cuda()
        trg = out_batch.cuda()
        trg_input = trg[:, :-1]
        trg_y = trg[:, 1:]

        max_target_length = trg_input.size(1)

        b_size = trg_input.size(0)

        src_mask, trg_mask = create_masks(src, trg_input, self)

        n_tokens = (trg_y != self.pad_tok).data.sum().items()

        encoder_op = self.encoder(src, src_mask)

        target_kb_mask = target_kb_mask.unsqueeze(-2)

        decoded_words = torch.zeros(b_size, int(max_target_length)).cuda()
        decoder_input = torch.zeros(b_size, int(max_target_length)).cuda().long()
        decoder_input[:, 0] = torch.LongTensor([self.sos_tok])
        all_decoder_outputs_vocab = Variable(torch.zeros(b_size, max_target_length, 512)).cuda()

        if kvl:
            subject_tracker = [[] for i in range(self.b_size)]

        for j in range(0, max_target_length):
            t_mask = trg_mask[:, :j + 1, :j + 1]

            if j == 0:
                decoder_input_zero = trg_input[:, 0:j + 1]
                decoder_op, decoder_self_attn = self.decoder(decoder_input_zero, encoder_op, src_mask, t_mask)
            else:
                decoder_op, decoder_self_attn = self.decoder(decoder_input[:, 0:j + 1], encoder_op, src_mask, t_mask)

            if kb_attn:
                kg_attn = self.kg_attention(decoder_self_attn, kb, target_kb_mask)
                decoder_op = decoder_op + kg_attn

            preds = self.generator(decoder_op)
            all_decoder_outputs_vocab[:, j] = decoder_op[:, -1, :]
            decoder_vocab = preds[:, -1, :]

            topv, topi = decoder_vocab.data.topk(1)

            if kvl:
                for k in range(self.b_size):
                    topi[k] = self.check_entity(topi[k].item(), kb[k], subject_tracker, k)

            if j != max_target_length - 1:
                bione = torch.zeros(b_size, int(max_target_length)).cuda().long()
                bione[:, j + 1] = Variable(topi.view(-1))
                decoder_input = decoder_input + bione
            decoded_words[:, j] = (topi.view(-1))

        loss = loss_compute(all_decoder_outputs_vocab, trg_y, n_tokens)

        batch_loss = loss / n_tokens

        del loss, src, trg, trg_input, trg_y, max_target_length, decoder_input, all_decoder_outputs_vocab, topi, topv
        torch.cuda.empty_cache()
        return decoded_words, batch_loss

    def evaluate_model(self, data, loss_compute, valid=False, test=False, kb_attn=True, kvl=True):
        if valid:
            batches = data.getBatches(self.b_size, valid=True, transpose=False)
        else:
            batches = data.getBatches(self.b_size, test=True, transpose=False)

        batches_len = 0
        for i, b in enumerate(batches):
            batches_len = i
            pass

        all_predicted = []
        target_batches = []

        f1_score = 0

        val_loss = 0
        for batch in batches:
            input_batch = Variable(torch.LongTensor(batch.encoderSeqs))
            out_batch = Variable(torch.LongTensor(batch.decoderSeqs))
            input_batch_mask = Variable(torch.FloatTensor(batch.encoderMaskSeqs))
            out_batch_mask = Variable(torch.FloatTensor(batch.decoderMaskSeqs))
            target_kb_mask = Variable(torch.LongTensor(batch.targetKbMask))

            kb = batch.kb_inputs

            decoded_words, loss_Vocab = self.evaluate_batch(input_batch, out_batch, input_batch_mask, out_batch_mask,
                                                            loss_compute, target_kb_mask=target_kb_mask, kb=kb,
                                                            kb_attn=kb_attn, kvl=kvl)

            batch_predictions = decoded_words.contiguous()

            val_loss += loss_Vocab

            f1_score += compute_f1(batch.targetSeqs, batch_predictions, self.entities_p.keys(), kb)

            all_predicted.append(batch_predictions)
            target_batches.append(batch.targetSeqs)

        candidates2, references2 = data.get_candidates(target_batches, all_predicted, True)

        moses_multi_bleu_score = moses_multi_bleu(candidates2, references2, True)

        if test:
            return moses_multi_bleu_score, val_loss / batches_len, candidates2, references2, f1_score / batches_len

        return moses_multi_bleu_score, val_loss / batches_len

    def check_entity(self, word, kb, subject_tracker, k):
        if word in self.entities_p.keys() and len(kb) > 1:
            for triple in kb:
                if self.entities_p[word] == triple[1] and triple[0] in subject_tracker[k]:
                    return triple[2]

                if word == triple[0] and word not in subject_tracker :
                    subject_tracker[k].append(word)
                    return word
                if word == triple[2] and word not in subject_tracker:
                    subject_tracker[k].append(word)
                    return word

        return word