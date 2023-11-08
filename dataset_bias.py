import json
import os
import random
from collections import defaultdict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.optim as Optim
import time
from torchvision.models.feature_extraction import create_feature_extractor
from biased_mnist import get_biased_mnist_dataloader
from bird.modules_finetune_bird import (
    LayerNorm,
    PositionalEmb,
    Transformer,
    decode_mask,
    subsequent_mask,
)
from torch.utils.data.dataloader import default_collate
from bird.para import parse_args
from tqdm import tqdm
from torchvision.models import ResNet101_Weights, resnet101

MASK = 5
BOS = 1


class PositionalEmb(nn.Module):
    "Implement the PE function for img or text or <CLS>."

    def __init__(self, d_model, dropout, max_len=7, max_seq_len=200):
        super(PositionalEmb, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.type_pe = torch.nn.Embedding(max_len, d_model)
        self.layernorm = LayerNorm(d_model, eps=1e-12)
        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, mode=None, pos=None):
        batchSize = x.size(0)
        patchLen = x.size(1)
        if mode in ["img1", "img2", "img"]:
            img1 = torch.LongTensor(batchSize, patchLen).fill_(pos)
            img_position = Variable(img1).cuda()
            type_pe = self.type_pe(img_position)
            pos_pe = Variable(self.pe[:, :patchLen], requires_grad=False).cuda()
            x = x + type_pe + pos_pe
        elif mode == "text":
            # make embeddings relatively larger
            x = x * math.sqrt(self.d_model)
            text = torch.LongTensor(batchSize, patchLen).fill_(pos)
            text_position = Variable(text).cuda()
            type_pe = self.type_pe(text_position)
            pos_pe = Variable(self.pe[:, :patchLen], requires_grad=False).cuda()
            x = x + type_pe + pos_pe
        else:  # [cls] or [diff] or [single]
            CLS = torch.LongTensor(batchSize, 1).fill_(pos)
            cls_type = Variable(CLS).cuda()
            type_pe = self.type_pe(cls_type)
            x = x + type_pe
        # layer norm & dropout
        x = self.layernorm(x)
        x = self.dropout(x)
        return x


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, args, data_path, vocabs, rev_vocabs, images, split, set_type=None
    ):
        # set parameter
        self.max_len = args.max_len
        self.vocabs = vocabs
        self.rev_vocabs = rev_vocabs
        self.split = split
        self.dataset = args.dataset
        self.set_type = set_type

        self.BOS = vocabs["<BOS>"]
        self.EOS = vocabs["<EOS>"]
        self.PAD = vocabs["<PAD>"]
        self.UNK = vocabs["<UNK>"]
        self.MASK = vocabs["<MASK>"]
        self.CLS = vocabs["<CLS>"]

        # load data
        if self.split == "train" or self.set_type == "P":
            self.load_data_multi_sents(data_path)
        else:
            self.load_data(data_path)

        self.images = {}
        for type in images.keys():
            self.images.update(images[type])

    def load_data_multi_sents(self, data_path):
        self.datas = []
        count_s = 0
        count_ns = 0
        with open(data_path, "r", encoding="utf-8") as fin:
            for line in fin:
                jterm = json.loads(line.strip())
                # change multi description to one description per data
                img2name = jterm["img2"]
                type = img2name.split("_")[1]
                if type == "semantic":
                    count_s += len(jterm["sentences"])
                else:
                    count_ns += len(jterm["sentences"])
                for des in jterm["sentences"]:
                    new_jterm = {}
                    new_jterm["img1"] = jterm["img1"]
                    new_jterm["img2"] = jterm["img2"]
                    new_jterm["sentences"] = des.split(" ")
                    self.datas.append(new_jterm)
                    if (
                        self.set_type == "P"
                    ):  # for efficient, validate one datat per caption
                        break
        print(
            self.split,
            "Total datas ",
            len(self.datas),
            "; semantic ",
            count_s,
            "; nonsemantic ",
            count_ns,
        )

    def load_data(self, data_path):
        self.datas = []
        with open(data_path, "r", encoding="utf-8") as fin:
            for line in fin:
                jterm = json.loads(line.strip())
                jterm["sentences"] = [cap.split(" ") for cap in jterm["sentences"]]
                self.datas.append(jterm)
        print("Total datas ", len(self.datas))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        description = data["sentences"]
        batch = {}
        # train
        if self.split == "train" or self.set_type == "P":
            # get raw triplet input data (img1, img2, text)
            img1 = torch.from_numpy(self.images[data["img1"]]).float()
            img2 = torch.from_numpy(self.images[data["img2"]]).float()
            dim, n, n = img1.size(0), img1.size(1), img1.size(2)
            img1, img2 = img1.view(dim, -1).transpose(0, 1), img2.view(
                dim, -1
            ).transpose(0, 1)
            # make sure img1 & img2 shape = [49,2048]
            ImgId = data["img1"] + "_" + data["img2"]
            # semantic or non semantic change
            type = data["img2"].split("_")[1]
            if type == "semantic":
                diff_label = 1
            else:  # type = nonsemantic
                diff_label = 0
            cap, cap_len, cap_label = self.padding(description)
            batch["img1"] = img1
            batch["img2"] = img2
            batch["cap"] = cap
            batch["cap_label"] = cap_label
            batch["diff_label"] = torch.LongTensor([diff_label])
            return batch
        # valid and test (used for Inference algorithm)
        else:
            # get raw triplet input data (img1, img2, text)
            img1 = torch.from_numpy(self.images[data["img1"]]).float()
            img2 = torch.from_numpy(self.images[data["img2"]]).float()
            dim, n, n = img1.size(0), img1.size(1), img1.size(2)
            img1, img2 = img1.view(dim, -1).transpose(0, 1), img2.view(
                dim, -1
            ).transpose(0, 1)
            # make sure img1 & img2 shape = [49,2048]
            ImgId = data["img1"] + "_" + data["img2"]
            gt_caps = [" ".join(tokens) for tokens in description]
            return img1, img2, gt_caps, ImgId

    def padding(self, sent):
        if len(sent) > self.max_len - 3:
            sent = sent[: self.max_len - 3]
        text = list(map(lambda t: self.vocabs.get(t, self.UNK), sent))
        text, output_label = self.mask_sent(text)

        prob = random.random()
        if prob < 0.15:  # 15% mask <EOS>
            text = [self.BOS] + text + [self.MASK]
            output_label = [-1] + output_label + [self.EOS]
        else:
            text = [self.BOS] + text + [self.EOS]
            output_label = [-1] + output_label + [-1]
        length = len(text)
        text = text + [self.PAD] * (self.max_len - length)
        output_label = output_label + [-1] * (self.max_len - length)
        T = torch.LongTensor(text)
        output_label = torch.LongTensor(output_label)
        return T, length, output_label

    def random_mask(self, x, i, prob):
        # 80% randomly change token to mask token
        if prob < 0.8:
            x[i] = self.MASK
        # 10% randomly change token to random token
        elif prob < 0.9:
            x[i] = random.choice(list(range(len(self.vocabs))))
        # -> rest 10% randomly keep current token
        return x

    def mask_sent(self, x):
        output_label = []
        for i, token in enumerate(x):
            prob = random.random()
            # mask normal token with 15% probability
            if prob < 0.15:
                prob /= 0.15
                x = self.random_mask(x, i, prob)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
        if all(o == -1 for o in output_label):
            # at least mask 1
            output_label[0] = x[0]
            x[0] = self.MASK
        return x, output_label

    def CLS(self):
        return self.CLS


def get_dataset(args):
    return DatasetBias(
        args,
        data_path=None,
        vocabs=None,
        rev_vocabs=None,
        images=None,
        split=None,
        set_type=None,
    )


def get_dataloader(input_dataset, batch_size, is_train=True, self_define=False):
    # use torch default dataloader function
    if not self_define:
        loader = torch.utils.data.DataLoader(
            dataset=input_dataset, batch_size=batch_size, shuffle=is_train
        )
    # use self defined dataloader func to get gts/imgID list batch
    else:
        loader = DataLoader(
            dataset=input_dataset, batch_size=batch_size, shuffle=is_train
        )
    return loader


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        kwargs["collate_fn"] = self_collate
        super().__init__(dataset, **kwargs)


import torchsnapshot
import tensordict


class DatasetBias(Dataset):
    def __init__(
        self, args, data_path, vocabs, rev_vocabs, images, split, set_type=None
    ):
        # set parameter
        self.max_len = args.max_len
        self.vocabs = vocabs
        self.rev_vocabs = rev_vocabs
        self.split = "test"
        self.dataset = args.dataset
        self.set_type = None

        # self.BOS = vocabs["<BOS>"]
        # self.EOS = vocabs["<EOS>"]
        # self.PAD = vocabs["<PAD>"]
        # self.UNK = vocabs["<UNK>"]
        # self.MASK = vocabs["<MASK>"]
        # self.CLS = vocabs["<CLS>"]

        # load data

        self.load_data(dataset="BiasedMNIST")

        # self.images = {}
        # for type in images.keys():
        #    self.images.update(images[type])

    def load_data_multi_sents(self, data_path):
        self.datas = []
        count_s = 0
        count_ns = 0
        with open(data_path, "r", encoding="utf-8") as fin:
            for line in fin:
                jterm = json.loads(line.strip())
                # change multi description to one description per data
                img2name = jterm["img2"]
                type = img2name.split("_")[1]
                if type == "semantic":
                    count_s += len(jterm["sentences"])
                else:
                    count_ns += len(jterm["sentences"])
                for des in jterm["sentences"]:
                    new_jterm = {}
                    new_jterm["img1"] = jterm["img1"]
                    new_jterm["img2"] = jterm["img2"]
                    new_jterm["sentences"] = des.split(" ")
                    self.datas.append(new_jterm)
                    if (
                        self.set_type == "P"
                    ):  # for efficient, validate one datat per caption
                        break
        print(
            self.split,
            "Total datas ",
            len(self.datas),
            "; semantic ",
            count_s,
            "; nonsemantic ",
            count_ns,
        )

    @torch.no_grad()
    def load_data(self, dataset):
        if dataset == "BiasedMNIST":
            dict_path = (
                "/home/infres/rnahon/projects/IDC/data_MNIST/dict_resnet101_test"
            )
            dataset_length = 600
            bs = 100
            dict_bs = int(np.ceil(dataset_length / bs))
            self.datas = tensordict.TensorDict({}, [dict_bs])
            if os.path.exists(dict_path):
                snapshot = torchsnapshot.Snapshot(path=dict_path)
                # cf https://pytorch.org/tensordict/saving.html
                app_state = {
                    "state": torchsnapshot.StateDict(
                        tensordict=self.datas.state_dict(keep_vars=True)
                    )
                }
                x = self.datas["label1"]
                snapshot.restore(app_state=app_state)
            else:
                dl_aligned = get_biased_mnist_dataloader(
                    root="/home/infres/rnahon/projects/IDC/data_MNIST",
                    batch_size=bs,
                    data_label_correlation=0.99,
                )
                dl_balanced = get_biased_mnist_dataloader(
                    root="/home/infres/rnahon/projects/IDC/data_MNIST",
                    batch_size=bs,
                    data_label_correlation=0.10,
                )

                # Load ResNet101 - Pretrained on MNIST
                print("Loading model")
                weights = ResNet101_Weights.IMAGENET1K_V2
                preprocess = weights.transforms()
                model = resnet101(weights=weights).cuda()
                return_layer = {"layer4.2.relu_2": "bot"}
                # get the output of the bottleneck layer
                model_bot = create_feature_extractor(model, return_layer).cuda()
                model_bot.eval()
                # Iterate on both dataloaders at once
                dataloader_iterator = iter(dl_aligned)
                for i, (img2, label2, bias_label2, idx2) in enumerate(dl_balanced):
                    try:
                        (img1, label1, bias_label1, idx1) = next(dataloader_iterator)
                    except StopIteration:
                        dataloader_iterator = iter(dl_aligned)
                        (img1, label1, bias_label1) = next(dataloader_iterator)
                    # Resize img1, img2 to have them be 3x224x224 and fit to resnet101
                    if i < dict_bs:  # to remove
                        self.datas[i]["img1"] = model_bot(preprocess(img1.cuda()))[
                            "bot"
                        ].cpu()
                        self.datas[i]["label1"] = label1
                        self.datas[i]["bias_label1"] = bias_label1
                        self.datas[i]["img2"] = model_bot(preprocess(img2.cuda()))[
                            "bot"
                        ].cpu()
                        self.datas[i]["label2"] = label2
                        self.datas[i]["bias_label2"] = bias_label2
                print(self.datas[i]["label1"])
                app_state = {
                    "state": torchsnapshot.StateDict(
                        tensordict=self.datas.state_dict(keep_vars=True)
                    )
                }
                snapshot = torchsnapshot.Snapshot.take(
                    app_state=app_state, path=dict_path
                )

                self.datas2 = tensordict.TensorDict({}, [dict_bs])

                snapshot2 = torchsnapshot.Snapshot(path=dict_path)
                # cf https://pytorch.org/tensordict/saving.html
                app_state1 = {
                    "state": torchsnapshot.StateDict(
                        tensordict=self.datas2.state_dict(keep_vars=True)
                    )
                }
                snapshot2.restore(app_state=app_state1)
                print(f"datas label1= {self.datas['label1']}")
                print(f"datas2 label1= {self.datas2['label1']}")

                assert (self.datas == self.datas2).all()
                assert self.datas["img1"].batch_size == self.datas2["img1"].batch_size

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        # description = data['sentences']
        # batch = {}
        # train
        # if self.split == 'train' or self.set_type == 'P':
        # get raw triplet input data (img1, img2, text)
        # img1 = data["img1"]
        # img2 = data["img2"]
        # dim, n, n = img1.size(0), img1.size(1), img1.size(2)
        # img1, img2 = img1.view(dim, -1).transpose(0, 1), img2.view(dim, -1).transpose(
        #    0, 1
        # )
        ## make sure img1 & img2 shape = [49,2048]
        ## ImgId = data["img1"] + "_" + data["img2"]
        ## semantic or non semantic change
        ## type = data["img2"].split("_")[1]
        ## if type == "semantic":
        ##    diff_label = 1
        ## else:  # type = nonsemantic
        ##    diff_label = 0
        ## cap, cap_len, cap_label = self.padding(description)
        # batch["img1"] = img1
        # batch["img2"] = img2
        # batch["cap"] = None  # cap
        # batch["cap_label"] = None  # cap_label
        # batch["diff_label"] = None  # torch.LongTensor([diff_label])
        # return batch
        # valid and test (used for Inference algorithm)
        # else:
        #    # get raw triplet input data (img1, img2, text)
        img1 = data["img1"]
        img2 = data["img2"]
        y1 = data["label1"]
        y2 = data["label2"]
        b1 = data["bias_label1"]
        b2 = data["bias_label2"]
        dim, n, n = img1.size(0), img1.size(1), img1.size(2)
        ##print(img1)
        # img1, img2 = img1.view(dim, -1).transpose(0, 1), img2.view(dim, -1).transpose(
        #    0, 1
        # )
        ##print(img1)
        # make sure img1 & img2 shape = [49,2048]
        ImgId = None  # data['img1']+'_'+data['img2']
        gt_caps = None  #  [' '.join(tokens) for tokens in description]
        return img1, img2, y1, y2, b1, b2

    def padding(self, sent):
        if len(sent) > self.max_len - 3:
            sent = sent[: self.max_len - 3]
        text = list(map(lambda t: self.vocabs.get(t, self.UNK), sent))
        text, output_label = self.mask_sent(text)

        prob = random.random()
        if prob < 0.15:  # 15% mask <EOS>
            text = [self.BOS] + text + [self.MASK]
            output_label = [-1] + output_label + [self.EOS]
        else:
            text = [self.BOS] + text + [self.EOS]
            output_label = [-1] + output_label + [-1]
        length = len(text)
        text = text + [self.PAD] * (self.max_len - length)
        output_label = output_label + [-1] * (self.max_len - length)
        T = torch.LongTensor(text)
        output_label = torch.LongTensor(output_label)
        return T, length, output_label

    def random_mask(self, x, i, prob):
        # 80% randomly change token to mask token
        if prob < 0.8:
            x[i] = self.MASK
        # 10% randomly change token to random token
        elif prob < 0.9:
            x[i] = random.choice(list(range(len(self.vocabs))))
        # -> rest 10% randomly keep current token
        return x

    def mask_sent(self, x):
        output_label = []
        for i, token in enumerate(x):
            prob = random.random()
            # mask normal token with 15% probability
            if prob < 0.15:
                prob /= 0.15
                x = self.random_mask(x, i, prob)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
        if all(o == -1 for o in output_label):
            # at least mask 1
            output_label[0] = x[0]
            x[0] = self.MASK
        return x, output_label

    def CLS(self):
        return self.CLS


def self_collate(batch):
    transposed = list(zip(*batch))
    img1_batch = default_collate(transposed[0])
    img2_batch = default_collate(transposed[1])
    GT_batch = transposed[2]  # type: list
    ImgId_batch = transposed[3]  # type: list
    return (img1_batch, img2_batch, GT_batch, ImgId_batch)


class AveragingModel(nn.Module):
    # Model used to accumulate latent representations of images of both datasets, to produce only a difference between datasets
    def __init__(
        self,
        ff_dim,
        img_embs,
        n_hidden,
        n_head,
        n_block,
        se_block,
        de_block,
        vocab_size,
        dropout,
        max_len,
        CLS,
    ):
        super(AveragingModel, self).__init__()
        # initilize parameter
        self.ff_dim = ff_dim  # FeedForward 2048
        self.img_embs = img_embs  # 2048
        self.n_hidden = n_hidden  # 512
        self.n_head = n_head
        self.n_block = n_block
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dropout = dropout
        self.CLS = CLS

        self.word_embedding = nn.Embedding(self.vocab_size, n_hidden)
        self.img_project = nn.Sequential(nn.Linear(img_embs, n_hidden), nn.Sigmoid())
        self.position_encoding = PositionalEmb(n_hidden, dropout, 7)
        self.single_img_encoder = Transformer(
            n_embs=n_hidden,
            dim_ff=ff_dim,
            n_head=n_head,
            n_block=se_block,
            dropout=dropout,
        )
        self.double_img_encoder = Transformer(
            n_embs=n_hidden,
            dim_ff=ff_dim,
            n_head=n_head,
            n_block=de_block,
            dropout=dropout,
        )
        self.encoder = Transformer(
            n_embs=n_hidden,
            dim_ff=ff_dim,
            n_head=n_head,
            n_block=self.n_block,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(self.n_hidden, self.vocab_size)

    def forward(self, Img1, Img2, compute_loss=True, accumulation_dict=None):
        Img1 = Img1.cuda()
        Img2 = Img2.cuda()
        # convert token to word embedding & add position embedding
        text_embs = self.word_embedding(Cap)
        text_embs = self.position_encoding(text_embs, mode="text", pos=6)
        # reduce img feat dim & add img position embedding
        img_embs1 = self.position_encoding(self.img_project(Img1), mode="img1", pos=2)
        img_embs2 = self.position_encoding(self.img_project(Img2), mode="img2", pos=4)
        CLS = torch.tensor([[self.CLS]])
        CLS = CLS.repeat(Img1.size(0), 1)
        CLS = self.word_embedding(CLS.cuda())
        CLS1 = self.position_encoding(CLS, mode="cls", pos=1)
        CLS2 = self.position_encoding(CLS, mode="cls", pos=3)
        image_embs1 = torch.cat((CLS1, img_embs1), dim=1)
        image_embs2 = torch.cat((CLS2, img_embs2), dim=1)
        image_embs1 = self.single_img_encoder(image_embs1)
        image_embs2 = self.single_img_encoder(image_embs2)
        if accumulation_dict is not None:
            if accumulation_dict["nb_elts"] == 0:
                accumulation_dict["img1"] = img_embs1
                accumulation_dict["img2"] = img_embs2
                accumulation_dict["nb_elts"] = 1
            else:
                accumulation_dict["img1"] = (
                    accumulation_dict["nb_elts"] * accumulation_dict["img1"] + img_embs1
                ) / accumulation_dict["nb_elts"] + 1
                accumulation_dict["img2"] = (
                    accumulation_dict["nb_elts"] * accumulation_dict["img2"] + img_embs2
                ) / accumulation_dict["nb_elts"] + 1
                accumulation_dict["nb_elts"] = accumulation_dict["nb_elts"] + 1
            if accumulation_dict["return"]:
                return accumulation_dict
            else:
                img_embs1 = accumulation_dict["img1"]
                img_embs2 = accumulation_dict["img2"]
                image_embs = self.double_img_encoder(
                    torch.cat((image_embs1, image_embs2), dim=1)
                )
                # concate [cls,i mg1,img2,text] as input to Transformer
                input_embs = torch.cat((image_embs, text_embs), dim=1)
                # input_embs = torch.cat((CLS, img_embs1, img_embs2, text_embs), dim=1)
                img_toklen = Img1.size(1) + 1

                # input to Transformer
                att_mask = Variable(
                    subsequent_mask(Cap.size(0), img_toklen, Cap.size(1))
                ).cuda()
                output = self.encoder(input_embs, att_mask)
                text = output[:, (img_toklen * 2) :, :]
                # only compute masked tokens for better efficiency
                masked_output = (
                    text[Cap_label != -1].contiguous().view(-1, text.size(-1))
                )
                # map hidden dim to vocab size
                prediction_scores = self.output_layer(masked_output)

                return prediction_scores
        else:
            image_embs = self.double_img_encoder(
                torch.cat((image_embs1, image_embs2), dim=1)
            )
            # concate [cls,i mg1,img2,text] as input to Transformer
            input_embs = torch.cat((image_embs, text_embs), dim=1)
            # input_embs = torch.cat((CLS, img_embs1, img_embs2, text_embs), dim=1)
            self.img_toklen = Img1.size(1) + 1

            # input to Transformer
            att_mask = Variable(
                subsequent_mask(Cap.size(0), self.img_toklen, Cap.size(1))
            ).cuda()
            output = self.encoder(input_embs, att_mask)
            text = output[:, (self.img_toklen * 2) :, :]
            # only compute masked tokens for better efficiency
            masked_output = text[Cap_label != -1].contiguous().view(-1, text.size(-1))
            # map hidden dim to vocab size
            prediction_scores = self.output_layer(masked_output)
            if compute_loss:
                masked_lm_loss = F.cross_entropy(
                    prediction_scores,
                    Cap_label[Cap_label != -1].contiguous().view(-1),
                    reduction="none",
                )
                return masked_lm_loss, prediction_scores
            else:
                return prediction_scores

    @torch.no_grad()
    def greedy(self, Img1, Img2):
        # Inference algorithm
        # print("this is greedy() function")
        Img1 = Variable(Img1).cuda()
        Img2 = Variable(Img2).cuda()
        batch_size = Img1.size(0)
        img_embs1 = self.position_encoding(self.img_project(Img1), mode="img1", pos=2)
        img_embs2 = self.position_encoding(self.img_project(Img2), mode="img2", pos=4)
        CLS = torch.tensor([[self.CLS]])
        CLS = CLS.repeat(Img1.size(0), 1)
        CLS = self.word_embedding(CLS.cuda())
        CLS1 = self.position_encoding(CLS, mode="cls", pos=1)
        CLS2 = self.position_encoding(CLS, mode="cls", pos=3)
        image_embs1 = torch.cat((CLS1, img_embs1), dim=1)
        image_embs2 = torch.cat((CLS2, img_embs2), dim=1)
        image_embs1 = self.single_img_encoder(image_embs1)
        image_embs2 = self.single_img_encoder(image_embs2)
        image_embs = self.double_img_encoder(
            torch.cat((image_embs1, image_embs2), dim=1)
        )
        # to get img token k,v cache -> improve computation efficiency
        output = self.encoder(image_embs, step=0)
        # decoder process
        mask_token = Variable(
            torch.LongTensor([MASK] * batch_size).unsqueeze(1)
        ).cuda()  # MASK token: (batch size, 1)
        gen_tokens = Variable(
            torch.ones(batch_size, 1).long()
        ).cuda()  # input cap initialize: <BOS>
        for i in range(1, self.max_len):
            # each time input previous generated token + <MASK>
            Des = torch.cat([gen_tokens, mask_token], dim=-1)
            text_embs = self.word_embedding(Des)
            text_embs = self.position_encoding(text_embs, mode="text", pos=6)
            step_attn_mask = Variable(
                decode_mask(batch_size, Img1.size(1) + 1, i + 1), requires_grad=False
            ).cuda()
            out = self.encoder(text_embs, mask=step_attn_mask, step=i).squeeze()
            out = self.output_layer(out)
            prob = out[:, -1]  # output prob
            _, next_w = torch.max(prob, dim=-1, keepdim=True)
            next_w = next_w.data
            gen_tokens = torch.cat([gen_tokens, next_w], dim=-1)
        return gen_tokens

    @torch.no_grad()
    def greedy_from_embs(self, image_embs1, image_embs2):
        # Inference algorithm from embedding space
        batch_size = 1
        image_embs = self.double_img_encoder(
            torch.cat((image_embs1, image_embs2), dim=1)
        )
        # to get img token k,v cache -> improve computation efficiency
        output = self.encoder(image_embs, step=0)
        # decoder process
        mask_token = Variable(
            torch.LongTensor([MASK] * batch_size).unsqueeze(1)
        ).cuda()  # MASK token: (batch size, 1)
        gen_tokens = Variable(
            torch.ones(batch_size, 1).long()
        ).cuda()  # input cap initialize: <BOS>
        for i in range(1, self.max_len):
            # each time input previous generated token + <MASK>
            Des = torch.cat([gen_tokens, mask_token], dim=-1)
            text_embs = self.word_embedding(Des)
            text_embs = self.position_encoding(text_embs, mode="text", pos=6)
            step_attn_mask = Variable(
                decode_mask(batch_size, self.img_toklen + 1, i + 1), requires_grad=False
            ).cuda()
            out = self.encoder(text_embs, mask=step_attn_mask, step=i).squeeze()
            out = self.output_layer(out)
            prob = out[:, -1]  # output prob
            _, next_w = torch.max(prob, dim=-1, keepdim=True)
            next_w = next_w.data
            gen_tokens = torch.cat([gen_tokens, next_w], dim=-1)
        return gen_tokens


def test(args):
    test_set = get_dataset(args)
    test_loader = get_dataloader(
        test_set, batch_size=1, is_train=True, self_define=True  # 00,
    )
    # Prepare model
    model = AveragingModel(
        ff_dim=args.dim_ff,
        img_embs=args.img_embs,
        n_hidden=args.n_embs,
        n_head=args.n_head,
        n_block=args.n_block,
        se_block=args.se_block,
        de_block=args.de_block,
        vocab_size=5000,
        dropout=args.dropout,
        max_len=args.max_len,
        CLS=None,
    )

    # load checkpoint
    if args.restore != "":
        print("load parameters from {}".format(args.restore))
        checkpoint = torch.load(args.restore)
        if "model_state_dict" in checkpoint.keys():
            pretrained_dict = checkpoint["model_state_dict"]
        else:
            pretrained_dict = checkpoint
        model_dict = model.state_dict()
        print([k for k, v in pretrained_dict.items() if k not in model_dict])
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and "pooler" not in k
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("load checkpoint finished")

    model.cuda()
    model.eval()
    # Prepare optimizer

    tk0 = tqdm(
        test_loader,
        total=int(len(test_loader)),
        leave=True,
    )
    for step, batch in enumerate(tk0):
        tk0.set_postfix(batch=step)
        img1 = batch[0]
        print(torch.sum(img1))
        img2 = batch[1]
        y1 = batch[2]
        print(y1)

        y2 = batch[3]
        b1 = batch[4]
        b2 = batch[5]
        ids = model.greedy(img1, img2).data.tolist()
        # global_step += 1
        # model.eval()
        # model.zero_grad()
        # mlm_loss, itm_loss = model(batch, compute_loss=True)
        # mlm_loss = mlm_loss.sum()
        # itm_loss = itm_loss.sum()
        # if args.l2_wd > 0:
        #    l2_loss = L2_SP(args, model, pretrained_dict)
        #    loss = mlm_loss + itm_loss + l2_loss
        # else:
        #    loss = mlm_loss + itm_loss
        # loss.backward()
    #
    ## learning rate scheduling
    # lr_this_step = get_lr_sched(global_step, args, pretrain=False)
    # for param_group in optim.param_groups:
    #    param_group['lr'] = lr_this_step
    #
    ## learning rate keep unchanged lr = 5e-5
    # writer.add_scalar('lr', optim.state_dict()['param_groups'][0]['lr'], global_step)
    # optim.step()
    #
    # report_mlm_loss += mlm_loss.data.item()
    # report_itm_loss += itm_loss.data.item()
    # n_samples += len(batch['img1'].data)
    #
    ## evaluating and logging
    # if global_step > 0 and global_step % args.report == 0:
    #    # report train loss & writing to log & add to tensorboard
    #    print('global_step: %d, epoch: %d, report_mlm_loss: %.3f, report_itm_loss: %.3f, time: %.2f'
    #            % (global_step, global_step//len(train_loader), report_mlm_loss / n_samples, report_itm_loss / n_samples, time.time() - start_time))
    #    train_logger.print_train_stats('mlm', global_step//len(train_loader), global_step, [report_mlm_loss / n_samples, report_itm_loss / n_samples], time.time() - start_time, stage='finetune')
    #    writer.add_scalar(args.dataset + '_train/mlm_loss', report_mlm_loss/n_samples, global_step)
    #    writer.add_scalar(args.dataset + '_train/itm_loss', report_itm_loss/n_samples, global_step)
    #    # calculate validate loss + word acc
    #    stats = {}
    #    stats = validate(valid_loader_P, model, global_step)
    #    # Inference algorithm & calculate main metric
    #    scores = Inference(valid_loader, test_loader, model, global_step)
    #    score = scores[args.main_metric]
    #    model.train()
    #    if score > best_score:
    #        no_beat = 0
    #        best_score = score
    #        print('Score Beat ', score, '\n')
    #        save_model(os.path.join(checkpoint_path, 'best_checkpoint.pt'), model)
    #    else:
    #        no_beat += 1
    #        # save_model(os.path.join(checkpoint_path, 'checkpoint_{}.pt'.format(global_step)), model)
    #        print('Term ', no_beat, 'Best Term', best_score, '\n')
    #    # add main metric score to log
    #    stats.update(scores)
    #    stats['main_metric_best'] = best_score
    #    val_logger.print_eval_stats(global_step, stats, no_beat)
    #    for k,v in stats.items():
    #        writer.add_scalar(args.dataset + '_validate/' + k, v, global_step)
    #
    #    # early stop
    #    if no_beat == args.early_stop:
    #        test(test_loader)
    #        sys.exit()
    #    report_loss, start_time, n_samples = 0, time.time(), 0
    # if global_step > args.total_train_steps:
    #    test(test_loader)
    #    sys.exit()
    # print('Learning Rate ', optim.state_dict()['param_groups'][0]['lr'])
    return 0


if __name__ == "__main__":
    args = parse_args()
    test(args)
