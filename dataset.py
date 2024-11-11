import torch
import torch.functional as F
import json
import os
import random
import concurrent.futures
import random
import multiprocessing
from functools import partial
from tokenizer import MusicalTokenizer

class MusicalSequenceDataset():

    def __init__(self, dataset, seq_len=512, tokenize=None, device="cuda", min_track_length =512, val_proportion = 0.05, shard_range=[0,5], augmentation={'pitch_steps':2, 'time_coeffs':[0.95, 1.05]}):

        if type(dataset) == list:
            val_number = int(len(dataset)*val_proportion)
            self.training_set, self.validation_set = dataset[val_number:], dataset[:val_number]

        else:
            self.augmentation=augmentation
            if 1 not in self.augmentation['time_coeffs']:
                self.augmentation['time_coeffs'].append(1)
            self.shard_range = shard_range
            self.min_track_length = min_track_length
            self.device = device
            self.seq_len = seq_len
            self.is_val = False
            self.val_proportion = val_proportion
            self.tokenize = tokenize
            self.augmentation = augmentation

            self.training_set, self.validation_set = [], []

            self.load_from_directory(dataset)
            
    def augment(self, song):
        """
        returns a list of songs including original and augmented versions, a total of (pitch_steps*2+1) * (len(time_coeffs)+1)
        """
        versions=[]
        for time_coeff in self.augmentation['time_coeffs']:
            new_song = []
            for track in song:
                new_track = []
                for event in track:
                    new_track.append([event[0], event[1]*time_coeff])
                new_song.append(new_track)
            versions.append(new_song)
        return versions
                

    
    def load_from_directory(self, path):
        """
        Loads raw track dicts in parallel.
        """

        for file in os.listdir(path)[self.shard_range[0]:self.shard_range[1]]:
            file=os.path.join(path, file)
            shard=json.load(open(file,'r'))
            self.process_shard(shard)
            print("shard processed")

    def process_shard(self, shard):
        """
        Extracts an array of songs filtered by length, augments it, tokenizes and distributes to training and validation sets
        """
        songs = [] 
        for i, song in enumerate(shard):
            valid_tracks=[]
            print(i)
            #if i == 460:
            #    continue
            for track in song['tracks']:
                if len(track) > self.min_track_length:
                    valid_tracks.append(track)
                    
            if len(valid_tracks) < 1: continue

            songs.append([[self.tokenize(track) for track in variant] for variant in self.augment(valid_tracks)])
            
        random.shuffle(songs)
        
        val_number = int(len(songs)*self.val_proportion)
        self.training_set.extend(songs[val_number:])
        self.validation_set.extend(songs[:val_number])
        
    
    def __len__(self):
        return len(self.training_set) if not self.is_val else len(self.validation_set)


if __name__== '__main__':
    import sys
    from tokenizer import MusicalTokenizer
    tokenizer=MusicalTokenizer(notes=128, 
                           time_shifts=125, 
                           longest_time_token=1000)

    shard_range=[int(sys.argv[1]), int(sys.argv[2])]
    from dataset import MusicalSequenceDataset
    dtst=MusicalSequenceDataset("F:/datasets/lmd_processed", tokenize=tokenizer.tokenize, device='cpu', min_track_length =int(sys.argv[3]), 
                                shard_range=shard_range, 
                                augmentation={'pitch_steps':0, 'time_coeffs':[0.9, 1.1]},
                                val_proportion = 0.01)
    torch.save((dtst.training_set+dtst.validation_set), f"lakh_dataset{shard_range}.pt")