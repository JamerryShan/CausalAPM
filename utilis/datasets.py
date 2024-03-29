
import torch
from torch.utils.data import Dataset

import pandas as pd
import csv


SENTENCEs_MAX_LENGTH = 128
# MAX_INPUT_LENGTH = SENTENCEs_MAX_LENGTH * 2

def check_data(s1, s2, args):

    if len(s1.split()) == 0 or len(s2.split()) == 0:
        print('INFO: check_data error')
        return False
    else:
        return True

def cal_shadow_label(sent1, sent2, tokenizer):
    e1 = tokenizer.encode(sent1)
    e2 = tokenizer.encode(sent2)
    set1 = set()
    set2 = set()
    for i in e1:
        set1.add(i)
    for i in e2:
        set2.add(i)
    return float(
        len(set1 & set2) / max(len(set1), len(set2))
        )

def processSentences(tokenizer, samples1_list, samples2_list):
    input_ids = []
    attention_masks = []
    segment_ids = []
    shadow_targets = []

    if len(samples1_list) != len(samples2_list):
        raise AssertionError
    
    for idx in range(len(samples1_list)):
        sent1 = samples1_list[idx]
        sent2 = samples2_list[idx]
        encoding = tokenizer.encode_plus(sent1, sent2, max_length=SENTENCEs_MAX_LENGTH, truncation=True, padding='max_length')

        input_id = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        segment_id = encoding['token_type_ids']

        
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        segment_ids.append(segment_id)
        shadow_targets.append(cal_shadow_label(sent1, sent2, tokenizer))

    
    return input_ids, attention_masks, segment_ids, shadow_targets


# datasets(valdir, tokenizer, args.dataset, args)

class datasets(Dataset):

    def __init__(self, datadir, tokenizer, dataset_name, args):
        df_samples = pd.read_csv(datadir, sep='\t', engine='python', quoting=csv.QUOTE_NONE)
        self.samples1_list = []
        self.samples2_list = []
        self.samples_labels = []

        self.transdict = {'entailment': 1, 'contradiction': 0, 'neutral': 2, 'non-entailment': 0, 'hidden': 0}
        # similarity,sentence1,sentence2
        for _, row in df_samples.iterrows():
            if dataset_name == 'MNLI' or dataset_name == 'HANS':
                s1 = str(row['sentence1'])
                s2 = str(row['sentence2'])
                label = row['gold_label']
                label = self.transdict[label]
            elif dataset_name == 'FEVER' or dataset_name == 'SYMM':
                s1 = str(row['premise'])
                s2 = str(row['hypothesis'])
                label = int(row['label'])
            elif dataset_name == 'QQP' or dataset_name == 'PAWS':
                s1 = str(row['sentence1'])
                s2 = str(row['sentence2'])
                label = int(row['label'])
                
            if check_data(s1, s2, args):
                self.samples1_list.append(s1)
                self.samples2_list.append(s2)
                self.samples_labels.append(label)
        
        self.len = len(self.samples_labels)
        
        input_ids, attention_masks, segment_ids, shadow_targets = processSentences(tokenizer, self.samples1_list, self.samples2_list)
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.segment_ids = segment_ids
        self.shadow_targets = shadow_targets

    
    def __getitem__(self, index):
        
        # print(self.input_ids[index])

        return self.input_ids[index], self.attention_masks[index], self.segment_ids[index], self.samples_labels[index], self.shadow_targets[index]

    def __len__(self):
        return self.len



class Collate_function:
    def collate(self, batch):
        input_ids, attention_masks, segment_ids, targets, shadow_targets = zip(*batch)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        shadow_targets = torch.tensor(shadow_targets, dtype=torch.float64)
        

        return input_ids, attention_masks, segment_ids, targets, shadow_targets

    def __call__(self, batch):
        return self.collate(batch)