from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import sys
sys.path.insert(0,'/home/huda/CLIP4Clip')
import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
import pandas

from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from torch.utils.data import DataLoader
from dataloaders.rawvideo_util import RawVideoExtractor
import multiprocessing



class AVSD_Dataset(Dataset):
    """Charades dataset dataset"""
    def __init__(
            self,
            data_path,
            features_path,
            data_set='/home/huda/avsd_spring2022/data/avsd/valid_i3d_data.json',
            dataset_val='/home/huda/avsd_spring2022/data/avsd/val_options.json',
            #avsd_path='/home/huda/avsd_spring2022/data/avsd/train_data.json',
            tokenizer=None,
            opt=None,
            subset='train',
            max_words=32,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
            self.opt = opt
            self.data_path = data_path
            self.features_path = features_path
            self.feature_framerate = feature_framerate
            self.max_words = max_words
            self.max_frames = max_frames
            self.tokenizer = tokenizer
            # 0: ordinary order; 1: reverse order; 2: random order.
            self.frame_order = frame_order
            assert self.frame_order in [0, 1, 2]
            # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
            self.slice_framepos = slice_framepos
            assert self.slice_framepos in [0, 1, 2]

            self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
            self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                                  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

            self.subset = subset
            assert self.subset in ["train", "val", 'test']

            train_file = os.path.join(self.data_path, "train_i3d_data.json")
            val_file = os.path.join(self.data_path, "val_i3d_data.json")
            test_file = os.path.join(self.data_path, "test_data.json")
            video_train_file = os.path.join(self.data_path,"Charades/Charades_v1_train.csv")
            dailog_train_file = os.path.join(self.data_path,"train_data.json")
            dailog_val_file = os.path.join(self.data_path,"avsd_val_wt_options.json")
            
            if self.subset == 'train':
                self.data2 = pandas.read_csv(video_train_file)
                with open(train_file, "r") as fp:
                    self.data = json.load(fp)
                    self.sample_len = len(self.data)
                dialogs_data = json.load(open(dailog_train_file, "r"))
            else:
                self.data = json.load(open(dailog_val_file, "r"))
                self.sample_len = len(self.data)

    def __len__(self):
        return len(self.data)

    def _get_text_avsd(self, question, dialog_history):

        pairs_text = np.zeros((1, self.max_words), dtype=np.compat.long)
        pairs_mask = np.zeros((1, self.max_words), dtype=np.compat.long)
        pairs_segment = np.zeros((1, self.max_words), dtype=np.compat.long)

        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]]
        for dialog in dialog_history:
            words = words + self.tokenizer.tokenize(dialog) # + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        question = self.tokenizer.tokenize(question) + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        question_len = len(question)

        total_length_with_CLS_question = self.max_words - question_len - 1

        # in case text needs to be shortened, add SEP token at end (assumes last SEP token spliced out)
        if len(words) > total_length_with_CLS_question:
            words = words[:total_length_with_CLS_question]

        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        words = words + question

        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]

        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(segment_ids) == self.max_words

        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def _get_rawvideo(self, name, s, e):

        video_mask = np.zeros((len(s), self.max_frames), dtype=np.compat.long)
        max_video_length = [0] * len(s)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(s), self.max_frames, 1, 3,self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)

        #video_path = os.path.join(self.features_path + '/' + self.data['id'][name] + '.mp4')

        video_path = os.path.join(self.features_path + '/' + name + '.mp4')
        try:
            for i in range(len(s)):
                start_time = int(s[i])
                end_time = int(e[i])
                start_time = start_time if start_time >= 0. else 0.
                end_time = end_time if end_time >= 0. else 0.
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                elif start_time == end_time:
                    end_time = end_time + 1

                # Should be optimized by gathering all asking of this video
                raw_video_data = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time)
                raw_video_data = raw_video_data['video']

                if len(raw_video_data.shape) > 3:
                    raw_video_data_clip = raw_video_data
                    # L x T x 3 x H x W
                    raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                    if self.max_frames < raw_video_slice.shape[0]:
                        if self.slice_framepos == 0:
                            video_slice = raw_video_slice[:self.max_frames, ...]
                        elif self.slice_framepos == 1:
                            video_slice = raw_video_slice[-self.max_frames:, ...]
                        else:
                            sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                            video_slice = raw_video_slice[sample_indx, ...]
                    else:
                        video_slice = raw_video_slice

                    video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                    slice_len = video_slice.shape[0]
                    max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                    if slice_len < 1:
                        pass
                    else:
                        video[i][:slice_len, ...] = video_slice
                else:
                    print("video path: {} error. video id: {}, start: {}, end: {}".format(video_path, name, start_time, end_time))
        except Exception as excep:
            print("video path: {} error. video id: {}, start: {}, end: {}, Error: {}".format(video_path, name, s, e, excep))
            raise excep

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # if video_mask.sum() != 64:
        # 	set_trace()
        # 	print('25')
        #print('video length', raw_video_slice.shape[0])
        return video, video_mask

    def _get_text(self, video_id, sentence):

        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.compat.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.compat.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.compat.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1

            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def __getitem__(self, idx):

        sample = self.data[idx]
        question = sample['question']
        answer = sample['answer']
        dialog_history = sample['dialog_history']
        pairs_text, pairs_mask, pairs_segment = self._get_text_avsd(question, dialog_history)
        video_id = sample['vid_id']
        video, video_mask = self._get_rawvideo(video_id, [0], [99999])

        if self.subset == 'val':  # retrun all the options, and in the fly encode them
            options = sample['options']
            options_encode = (self._get_text(video_id, op) for op in options)
            return pairs_text, pairs_mask, pairs_segment, video, video_mask, options_encode

        return pairs_text, pairs_mask, pairs_segment, video, video_mask

if __name__ == "__main__":

    tokenizer = ClipTokenizer()
    data = AVSD_Dataset(tokenizer=tokenizer,subset='val')
    data_loader = DataLoader(data, batch_size=1, shuffle=False)
    next(iter(data_loader))
    for step, batch in enumerate(data_loader):
        print('step')