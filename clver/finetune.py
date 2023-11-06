'''
    Transformer based pretraining Model
'''

import os
import sys
import json
import time
import argparse
from collections import defaultdict
# from bert_score import BERTScorer

import torch
import torch.nn as nn
import torchvision
import torch.optim as Optim
from torch.autograd import Variable
import torch.nn.functional as F
from utils import *
import modules_finetune as modules
from eval import Evaluator

import pdb
import shutil
import random
from tqdm import tqdm
from shutil import copyfile

from para import parse_args


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import dataset: diff dataset use diff function
if args.dataset == 'bird':
    import data.dataset_finetune_bird as dataset
elif args.dataset == 'ImageEdit':
    import data.dataset_finetune_edit as dataset
elif args.dataset == 'clver':
    import data.dataset_finetune_clver as dataset
# use tensorboard
from torch.utils.tensorboard import SummaryWriter 
# random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# load images
images = load_split_images(args.data_path)
vocabs, rev_vocabs = load_vocabs(args.vocab_path)
vocab_size = len(vocabs)

if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
out_path = os.path.join('experiments', args.exp_name)
if not os.path.exists(out_path):
    os.mkdir(out_path)
checkpoint_path = os.path.join(out_path, 'checkpoint') 
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path) 

if args.mode == 'train':
    val_sent_path = os.path.join(out_path, 'val_sent') 
    if not os.path.exists(val_sent_path):
        os.mkdir(val_sent_path)  
    train_logger = Logger(out_path, is_train=True)
    val_logger = Logger(out_path, is_train=False)
    # save configs to json
    save_para(args, out_path + '/config.json')
    # set tensorboard log file
    if not os.path.exists(out_path+'/log/'):
        os.mkdir(out_path+'/log/')  
    writer = SummaryWriter(out_path+'/log/')

# load BERTScore model to calculate metric
# ref_sents = get_ref_sents(os.path.join(args.data_path, 'test.json'))
# scorer = BERTScorer(lang="en", rescale_with_baseline=True, idf=True, idf_sents=ref_sents, device='cuda:'+str(0))
def save_model(path, model):
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, path)

def get_dataset(data_path, images, split, set_type=None):
    return dataset.Dataset( args,
                            data_path,
                            vocabs = vocabs,
                            rev_vocabs = rev_vocabs,
                            images = images,
                            split = split,
                            set_type = set_type
                            )

def get_dataloader(input_dataset, batch_size, is_train=True, self_define=False):
    # use torch default dataloader function
    if not self_define:
        loader = torch.utils.data.DataLoader(dataset=input_dataset, batch_size=batch_size, shuffle=is_train)
    # use self defined dataloader func to get gts/imgID list batch
    else:
        loader = dataset.DataLoader(dataset=input_dataset, batch_size=batch_size, shuffle=is_train)
    return loader



def validate(loader, model, global_step):
    print('Starting Evaluation...')
    model.eval()
    mlm_loss = 0
    itm_loss = 0
    n_correct = 0
    n_word = 0
    tot_score = 0
    n_ex = 0 
    with torch.no_grad():
        for i, batch in enumerate(loader):
            mlm_scores, itm_scores = model(batch, compute_loss=False)
            # mlm loss
            cap_labels = Variable(batch['cap_label'], requires_grad=False).cuda()
            cap_labels = cap_labels[cap_labels != -1]
            loss = F.cross_entropy(mlm_scores, cap_labels, reduction='sum')
            mlm_loss += loss.item()
            n_correct += (mlm_scores.max(dim=-1)[1] == cap_labels).sum().item()
            n_word += cap_labels.numel()
            # itm loss
            itm_targets =  Variable(batch['diff_label'], requires_grad=False).cuda().view(-1)
            loss = F.cross_entropy(itm_scores, itm_targets, reduction='sum')
            itm_loss += loss.item()
            tot_score += (itm_scores.max(dim=-1)[1] ==itm_targets).sum().item()
            n_ex += itm_targets.size(0)
    mlm_loss /= n_word
    mlm_acc = n_correct / n_word
    itm_loss = itm_loss / n_ex
    itm_acc = tot_score / n_ex
    val_log = {'mlm loss': mlm_loss,
                'mlm acc': mlm_acc,
                'itm loss': itm_loss,
                'itm acc': itm_acc}
    return val_log

def Inference(loader, test_loader, model, global_step):
    print('Starting Inference...')
    start_time = time.time()
    model.eval()
    candidates = {}
    references = {}
    cand_lists = []
    ref_lists = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            img1, img2, gts, ImgID = batch
            ids = model.greedy(img1, img2).data.tolist()
            print(ids)
            for i in range(len(img1.data)):
                id = ids[i]
                sentences = transform(id)
                # show generate sent case
                if bi == 0 and i < 3:
                    print(' '.join(sentences))

                candidates[ImgID[i]] = [' '.join(sentences)]
                references[ImgID[i]] = gts[i]
                cand_lists.append(' '.join(sentences))
                ref_lists.append(gts[i])

    evaluator = Evaluator(references, candidates)
    score = evaluator.evaluate()
    # calculate BERTScore
    # P_mul, R_mul, F_mul = scorer.score(cand_lists, ref_lists)
    # score['BERTScore'] = F_mul.mean()

    # "ROUGE_L", "CIDEr"
    print(args.main_metric, score[args.main_metric])
    print("evaluting time:", time.time() - start_time)
    # save val generate sentences
    with open(os.path.join(val_sent_path, 'iter_{}.json'.format(global_step)), 'w', encoding='utf-8') as fout:
        for ImgID in candidates.keys():
            sample = {'ImgId': ImgID,
                      'candidates': candidates[ImgID]
                      # 'references': references[ImgID]
                    } 
            jterm = json.dumps(sample, ensure_ascii=False)
            fout.write(jterm + '\n')
    
    # evaluate for test
    # candidates = {}
    # references = {}
    # with torch.no_grad():
    #     for bi, batch in enumerate(test_loader):
    #         img1, img2, gts, ImgID = batch
    #         ids = model.greedy(img1, img2).data.tolist()
    #         for i in range(len(img1.data)):
    #             id = ids[i]
    #             sentences = transform(id)
    #             candidates[ImgID[i]] = [' '.join(sentences)]
    #             references[ImgID[i]] = gts[i]

    # evaluator = Evaluator(references, candidates)
    # test_score = evaluator.evaluate()
    # score['test_'+args.main_metric] = test_score[args.main_metric]
    return score



def test(test_loader=None, model=None):
    if test_loader is None:
        print("Test: load data ...")
        test_set = get_dataset(os.path.join(args.data_path, 'test.json'),
                                images, 'test')
        test_loader = get_dataloader(test_set, args.batch_size, is_train=False, self_define=True)
    if model is None:
        print("Test: load best checkpoint ...")
        # Prepare model
        model = modules.Model(ff_dim = args.dim_ff,
                            img_embs = args.img_embs,
                            n_hidden = args.n_embs,
                            n_head = args.n_head,
                            n_block = args.n_block,
                            img_enc_block = args.img_enc_block,
                            vocab_size = vocab_size,
                            dropout = args.dropout,
                            max_len = args.max_len,
                            CLS = vocabs['<CLS>'])

        if args.best_ckpt == '':
            print(" ==> test stage load checkpoint from ", os.path.join(out_path, 'checkpoint/best_checkpoint.pt'))
            model_dict = torch.load(os.path.join(out_path, 'checkpoint/best_checkpoint.pt'))
        else:
            print(" ==> test stage load checkpoint from ", args.best_ckpt)
            model_dict = torch.load(args.best_ckpt)
        model.load_state_dict({k.replace('module.', ''):v for k,v in model_dict.items()})
        model.cuda()

    model.eval()


    # def get_parameter_number(model):
    #     total_num = sum(p.numel() for p in model.parameters())
    #     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     return {'Total': total_num, 'Trainable': trainable_num}
    # para = get_parameter_number(model)
    # print(para)
    # print('# model parameters:', sum(param.numel() for param in model.parameters()))

    candidates = {}
    references = {}
    cand_lists = []
    ref_lists = []
    with torch.no_grad():
        with open(os.path.join(out_path, args.out_file), 'w', encoding='utf-8') as fout:
            for i, batch in enumerate(test_loader):
                img1, img2, _, _ = batch
                #print(img1)
                ids = model.greedy(img1, img2).data.tolist()
                for i in range(len(img1.data)):
                    id = ids[i]
                    sentences = transform(id)
                    #candidates[ImgID[i]] = [' '.join(sentences)]
                    #references[ImgID[i]] = gts[i]
                    #sample = {'ImgId': ImgID[i],
                    #        'candidates': [' '.join(s for s in sentences)]
                    #        'references': Caps
                    #        }
                    print(' '.join(sentences))
                    cand_lists.append(' '.join(sentences))
                    #ref_lists.append(gts[i])
                    #jterm = json.dumps(sample, ensure_ascii=False)
                    # fout.write(jterm + '\n')
    evaluator = Evaluator(references, candidates)
    score = evaluator.evaluate()
    # P_mul, R_mul, F_mul = scorer.score(cand_lists, ref_lists)
    # score['BERTScore'] = F_mul.mean()
    print(args.main_metric, score[args.main_metric])


def val():
    val_set = get_dataset(os.path.join(args.data_path, 'val.json'), images, 'val')
    val_loader = get_dataloader(val_set, args.batch_size, is_train=False, self_define=True)

    # Prepare model
    model = modules.Model(ff_dim = args.dim_ff,
                          img_embs = args.img_embs,
                          n_hidden = args.n_embs,
                          n_head = args.n_head,
                          n_block = args.n_block,
                          img_enc_block = args.img_enc_block,
                          vocab_size = vocab_size,
                          dropout = args.dropout,
                          max_len = args.max_len,
                          CLS = vocabs['<CLS>'])
    if args.best_ckpt == '':
        print(" ==> test stage load checkpoint from ", os.path.join(out_path, 'checkpoint/best_checkpoint.pt'))
        model_dict = torch.load(os.path.join(out_path, 'checkpoint/best_checkpoint.pt'))
    else:
        print(" ==> test stage load checkpoint from ", args.best_ckpt)
        model_dict = torch.load(args.best_ckpt)
    model.load_state_dict({k.replace('module.', ''):v for k,v in model_dict.items()})
    model.cuda()
    model.eval()
    candidates = {}
    references = {}
    with torch.no_grad():
        with open(os.path.join(out_path, "val_result.json"), 'w', encoding='utf-8') as fout:
            for bi, batch in enumerate(val_loader):
                img1, img2, gts, ImgID = batch
                ids = model.greedy(img1, img2).data.tolist()

                for i in range(len(img1.data)):
                    id = ids[i]
                    sentences = transform(id)

                    candidates[ImgID[i]] = [' '.join(sentences)]
                    references[ImgID[i]] = gts[i]

                    sample = {'ImgId': ImgID[i],
                            'candidates': [' '.join(s for s in sentences)]
                            #'references': Caps
                            }
                    
                    jterm = json.dumps(sample, ensure_ascii=False)
                    fout.write(jterm + '\n')
    evaluator = Evaluator(references, candidates)
    score = evaluator.evaluate()
    print(args.main_metric, score)

def transform(ids):
    sentences = []
    for wid in ids:
        if wid == vocabs['<BOS>']:
            continue
        if wid == vocabs['<EOS>']:
            break
        sentences.append(rev_vocabs[wid])
    return sentences


if __name__ == '__main__':
    if args.mode == 'train':
        print('------Train Mode------')
        train()
        test()
    elif args.mode == 'test':
        print('------Test  Mode------')
        test()
    elif args.mode == 'val':
        print('------Valid  Mode------')
        val()
    else:
        print('Please enter the mode')