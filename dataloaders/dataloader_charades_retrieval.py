from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

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

class Charades_Dataset(Dataset):
    """Charades dataset dataset"""
    def __init__(
            self,
            data_path='/data/ECLIPSE/charades/Charades',
            features_path='/data/ECLIPSE/charades/Charades_v1_480',
            #avsd_path='/home/huda/avsd_spring2022/data/avsd/train_data.json',
            tokenizer=None,
            opt=None,
            subset='train',
            max_words=30,
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
            self.data = pandas.read_csv('/data/ECLIPSE/charades/Charades/Charades_v1_' + self.subset + '.csv')
            '''
            dialogs = json.load(open("/home/huda/avsd_spring2022/data/avsd/train_data.json", "r"))
            data_dic = {}
            # loop through each dictionary in the array
            for d in data_arr:
                # get the vid_id key value
                vid_id = d['vid_id']
                # if the vid_id key already exists in the dictionary, append the dictionary to the existing list
                if vid_id in data_dic:
                    data_dic[vid_id].append(d)
                # otherwise, create a new list for the vid_id key and add the dictionary to it
                else:
                    data_dic[vid_id] = [d]

            # print the resulting dictionary
            print(data_dic)
            '''
    def __len__(self):
        return len(self.data)

    def _get_rawvideo(self, video_id, choice_video_ids=[1]):


        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.compat.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path+'/' + self.data['id'][video_id] + '.mp4')

            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
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
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def _get_rawvideo2(self, name, s, e):

        video_mask = np.zeros((len(s), self.max_frames), dtype=np.compat.long)
        max_video_length = [0] * len(s)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(s), self.max_frames, 1, 3,self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)

        video_path = os.path.join(self.features_path + '/' + self.data['id'][name] + '.mp4')

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

    def _get_text_dialog(self, video_id, sentence):

        question, dialog_history = sentence
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        pairs_text = np.zeros((k, self.max_words), dtype=np.compat.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.compat.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.compat.long)

        for i, video_id in enumerate(choice_video_ids):
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]]

            for dialog in dialog_history:
                words = words + self.tokenizer.tokenize(dialog) + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            question = self.tokenizer.tokenize(question) + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            question_len = len(question)

            total_length_with_CLS_question = self.max_words - question_len - 1

            if len(words) > total_length_with_CLS_question:
                words = words[:total_length_with_CLS_question]
                words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            words = words + question
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

        # k = n_caption
        # pairs_text = np.zeros((k, self.max_words), dtype=np.compat.long)
        # pairs_mask = np.zeros((k, self.max_words), dtype=np.compat.long)
        # pairs_segment = np.zeros((k, self.max_words), dtype=np.compat.long)

    '''
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
        '''
    def __getitem__(self, idx):

        video_id = self.data['id'][idx]
        caption = self.data['script'][idx]
        pairs_text, pairs_mask, pairs_segment= self._get_text(video_id,caption)
        video, video_mask = self._get_rawvideo2(idx, [0], [99999])

        return pairs_text, pairs_mask, pairs_segment, video, video_mask


if __name__ == "__main__":
    tokenizer = ClipTokenizer()
    data_test = Charades_Dataset(tokenizer=tokenizer)
    data_test_loader = DataLoader(data_test, batch_size=1, shuffle=True)
    next(iter(data_test_loader))
    print('test')
