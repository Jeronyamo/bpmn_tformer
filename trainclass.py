import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


PAD_TOKEN = 0

##  ============  Transformer hyperparameters  ============  ##

class TFormerParameters:
    def __init__(self, l_url, l_tt, l_m):
        self.max_seq_length = 512
        self.num_layers = 6
        self.num_heads = 8
        self.d_model = 512
        self.dropout = 0.1
        self.d_ff = 2048
        self.src_vocab_size = len(l_url) * len(l_tt) * len(l_m) + 5
        self.tgt_vocab_size = len(l_url) * len(l_tt) * len(l_m) + 5


##  ============  Data tokenizer  ============  ##

class TFormerTokenizer:
    """
    Receives a path to the options (predictable parameters, URL/TASKTYPE/METHOD).\n
    Tokenizes/detokenizes tasks.
    """
    def __init__(self, paramfile: str):
        self.paramfile = paramfile
        self.URL, self.TASKTYPE, self.METHOD = TFormerTokenizer.get_options(paramfile)
        self.l_url = len(self.URL)
        self.l_tt  = len(self.TASKTYPE)
        self.l_m   = len(self.METHOD)


    ##  Read options from "unique_params.txt"
    def get_options(fname: str) -> tuple[list[str], list[str], list[str]]:
        _URLs: list[str] = []; _TASKTYPEs: list[str] = []; _METHODs: list[str] = []
        tmp:   list[str] = []

        with open(fname) as params:
            for string in params.readlines():
                string = string.strip()
                if string == "URLs:":
                    tmp = _URLs
                    continue
                if string == "Task types:":
                    tmp = _TASKTYPEs
                    continue
                if string == "Methods:":
                    tmp = _METHODs
                    continue
                tmp.append(string)
        return _URLs, _TASKTYPEs, _METHODs

    ##  Encode: (URL, TaskType, Method) -> token
    def TaskEncode(self, l_list: tuple[int, int, int]) -> int:
        return l_list[0] + l_list[1] * self.l_url + l_list[2] * (self.l_url * self.l_tt)

    ##  Decode: token -> (URL, TaskType, Method)
    def TaskDecode(self, _num: int) -> tuple[int, int, int]:
        url_id = _num % self.l_url
        met_id = _num // (self.l_tt * self.l_url)
        tsk_id = (_num - url_id - met_id * (self.l_tt * self.l_url)) // self.l_url
        # return self.URL[url_id], self.TASKTYPE[tsk_id], self.METHOD[met_id] # tuple[str, str, str]
        return url_id, tsk_id, met_id

    def proc_decode(self, _proc: list[tuple[int]]) -> list[tuple[int, int, int]]:
        return [self.TaskDecode(_task) for _task in _proc]


##  ============  Data loader  ============  ##

class TFormerDataLoader:
    """
    Receives a path to the file containing encoded processes
    and process padded length. Generates batches for the transformer.
    """
    def __init__(self, procfile: str, pad_to_len: int):
        from random import shuffle

        self.procfile = procfile
        self.pad_to_len = pad_to_len

        train_data: list[tuple[int]] = TFormerDataLoader.get_procs_from_file(procfile, pad_to_len)
        shuffle(train_data)

        self.train_data: list[tuple[int]] = train_data[:int(0.8 * len(self.train_data)) ]
        self.  val_data: list[tuple[int]] = train_data[ int(0.8 * len(self.train_data)): int(0.9 * len(self.train_data))]
        self. test_data: list[tuple[int]] = train_data[ int(0.9 * len(self.train_data)):]


    ##  Read encoded processes from "processes.txt"
    def get_procs_from_file(fname: str, pad_to_len: int) -> list[tuple[int]]:
        proc_list: list[tuple[int]] = []
        proc_max_len = pad_to_len

        with open(fname) as procs:
            for line in procs.readlines():
                if (l := line.strip()) != "":
                    tmp = l.split(", ")
                    tmp.extend((PAD_TOKEN for _ in range(pad_to_len - len(tmp))))
                    tmp = tuple(int(task) for task in tmp)
                    if  proc_max_len < len(tmp):
                        proc_max_len = len(tmp)
                    proc_list.append(tmp)

        if proc_max_len > pad_to_len:
            raise RuntimeError("Error: process reader - there is a process with " + str(proc_max_len) + " tasks")
        return proc_list

    ##  Dataset generator
    def gen_dataset(self, ds_type: str, batch_size: int, start_from_batch: int = 0, to_batch: int|None = None):
        def DatasetGenWholeProc(procs: list[tuple[int]], batch_size: int, start_from_batch: int = 0, to_batch: int|None = None):
            tmp1, tmp2, tmp3 = [], [], []

            if to_batch is not None:
                to_batch *= batch_size
    
            for proc in procs[batch_size * start_from_batch : to_batch]:
                proc = list(proc)

                tmp1.append(proc)
                tmp2.append(proc[ :-1] + [PAD_TOKEN,])
                tmp3.append(proc[1:  ] + [PAD_TOKEN,])

                if len(tmp1) >= batch_size:
                    yield torch.tensor(tmp1, dtype= torch.long),\
                          torch.tensor(tmp2, dtype= torch.long),\
                          torch.tensor(tmp3, dtype= torch.long)
                    tmp1, tmp2, tmp3 = [], [], []

            if len(tmp1):
                yield torch.tensor(tmp1, dtype= torch.long),\
                      torch.tensor(tmp2, dtype= torch.long),\
                      torch.tensor(tmp3, dtype= torch.long)

        _procs = None
        if (ds_type == "train"): _procs = self.train_data
        if (ds_type ==   "val"): _procs = self.val_data
        if (ds_type ==  "test"): _procs = self.test_data
        if _procs is None: raise ValueError("No dataset of type " + str(ds_type) + '\n')

        return DatasetGenWholeProc(batch_size, start_from_batch, to_batch)


##  ============  Transformer model  ============  ##

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask= None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model= x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k= x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask= None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


def def_tformer_loader(_par: TFormerParameters):
    return Transformer(_par.src_vocab_size, _par.tgt_vocab_size, _par.d_model,
                       _par.num_heads, _par.num_layers, _par.d_ff, _par.max_seq_length, _par.dropout)

def def_tformer_ckpt(fname: str) -> Transformer:
    return torch.load(fname)


##  ============  Transformer trainer class  ============  ##

"""
Used for training
"""
class TFormerTrainer:
    def __init__(self, tformer: Transformer, _par: TFormerParameters, dloader: TFormerDataLoader, _tokenizer: TFormerTokenizer):
        self.model = tformer
        self.dloader = dloader
        self.tokenizer = _tokenizer
        self.params = _par
        self.loss = nn.CrossEntropyLoss(ignore_index= 0)
        self.optimizer = optim.Adam(self.model.parameters(), lr= 1e-4, betas= (0.9, 0.98), eps= 1e-9)


    def train(self, num_epochs: int,
                batch_size: int, from_batch: int = 0, to_batch: int|None = None) -> None:
        val_batch = 0
        self.model.train()

        print("Total number of trainable parameters =",
                sum((p.numel() for p in self.model.parameters() if p.requires_grad)))
        for epoch in range(num_epochs):
            i = from_batch
            for source, t_target, l_target in self.dloader.gen_dataset("train", batch_size, i, to_batch):
                self.optimizer.zero_grad()

                output = self.model(source, t_target)
                loss = self.loss(output.contiguous().view(-1, self.tokenizer.tgt_vocab_size), l_target.contiguous().view(-1))
                loss.backward()
                self.optimizer.step()

                val_every = 25
                val_num_batches = 10
                print("Iteration", i, f": loss = {loss.item()}", end= '\r')
                if i and not i % val_every:
                    print()
                    self.validation(batch_size, val_batch, val_batch + val_num_batches)
                    self.model.train()
                i += 1

    def validation(self, batch_size: int, from_batch: int = 0, to_batch: int|None = None) -> None:
        def get_topk_acc(_output: torch.Tensor, _target, k: list) -> list[int]:
            tokens = np.argsort(_output.numpy(), 2)[:, :, -k[0]:]
            s1 = [0 for _ in range(len(k))]

            for n_elem in range(tokens.shape[0]):
                s2 = [0 for _ in range(len(k))]

                x = list(tokens[n_elem])
                labels = _target[n_elem].tolist()

                for l in range(self.params.max_seq_length):
                    if not labels[l]:
                        for j in range(len(k)):
                            s1[j] += s2[j] / l
                        break

                    for j in range(len(k)):
                        s2[j] += labels[l] in x[l][-k[j]:]

            for j in range(len(k)):
                s1[j] /= tokens.shape[0]
            return s1


        self.model.eval()
        topk_accs = (5, 3, 1)

        with torch.no_grad():
            i, acc = from_batch, [0, 0, 0]
            for source, t_target, l_target in self.dloader.gen_dataset("val", batch_size, i, to_batch):
                self.optimizer.zero_grad()

                output = self.model(source, t_target)
                tmp = get_topk_acc(output, l_target, topk_accs)

                for j in range(len(topk_accs)):
                    acc[j] += tmp[j]
                i += 1
                print(f"Validation iter {i}: top-1 = {acc[2] / i:1.7}, top-3 = {acc[1] / i:1.7}, top-5 = {acc[0] / i:1.7}", end= '\r')
            print()



##  ============  Transformer handler class  ============  ##

"""
Used for prediciton
"""
class TFormerHandler:
    def __init__(self, tformer: Transformer):
        tformer.requires_grad_(False)
        self.model = tformer

    def predict(self, proc_in):
        pass