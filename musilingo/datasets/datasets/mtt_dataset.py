from musilingo.utils.lib import *
from musilingo.datasets.datasets.base_dataset import BaseDataset
from huggingface_hub import login
from datasets import concatenate_datasets

from collections import Counter

from datasets import load_dataset
import torch
import numpy as np
import torchaudio.transforms as T



RESAMPLE_RATE=24000


class MTTDataset(BaseDataset):
    def __init__(self, processor, split, genre, pct_other_genres, tf_idf, unit_type, top_k=-1, sr=16000, duration=29.124, arch="musilingo"):
        self.split = split
        self.n_samples = int(sr * duration)
        self.mtt_dataset = load_dataset("AndreiBlahovici/LP-MusicCaps-MTT", split=split)
        self.genre = genre
        self.top_k = top_k
        self.tf_idf = tf_idf
        self.unit_type = unit_type
        self.pct_other_genres = pct_other_genres
        self.arch = arch
        self.resample_rate = processor.sampling_rate
        self.processor = processor

        if self.genre != "all":
            focus_songs = self.mtt_dataset.filter(lambda x: x['genre'] == genre)

            if self.split == "train":
                other_songs = self.mtt_dataset.filter(lambda x: x['genre'] != genre)     

                sample_size = int(len(other_songs) * pct_other_genres)
                sampled_other_genres_songs = other_songs.shuffle(seed=42).select(range(sample_size))

                self.mtt_dataset = concatenate_datasets([focus_songs, sampled_other_genres_songs])
            else:
                self.mtt_dataset = focus_songs

        if self.arch == "musillm" or self.genre == "all":
            self.instruction = f"Write a caption for the provided music clip. The caption should be a complete sentence that describes the music clip. The caption should be in English and should not contain any profanity or hate speech."
        else:
            with open(f"genres_vocab/{genre}_vocab_{self.unit_type}_tf_idf_{self.tf_idf}.json", "r") as f:
                counts_vocabulary = json.load(f)

            self.vocabulary = [word for word, _ in Counter(counts_vocabulary).most_common(None if self.top_k == -1 else self.top_k)]

            self.instruction = f"Write a caption for the provided music clip. The caption should be a complete sentence that describes the music clip. The caption should be in English and should not contain any profanity or hate speech. The current song is from the {self.genre} genre. To generate the caption, make use of the following vocabulary{' of word pairs' if self.unit_type == 'bigram' else ''} that is commonly used to describe the music from this genre, but make sure to use other words too: " + ", ".join(self.vocabulary)


    def load_audio(self, index):
        audio = self.mtt_dataset[index]["audio"]["array"]

        resampler = T.Resample(self.mtt_dataset[index]["audio"]["sampling_rate"], self.resample_rate)
        audio_input = resampler(torch.from_numpy(audio).float())
        
        audio = self.processor(audio_input, 
                               sampling_rate=self.resample_rate, 
                               return_tensors="pt")['input_values'][0]
        
        # exit(0)

        # if audio.shape[-1] < self.n_samples:
        #     pad = np.zeros(self.n_samples)
        #     pad[:audio.shape[-1]] = audio
        #     audio = pad

        # random_idx = random.randint(0, audio.shape[-1]-self.n_samples)
        # audio_tensor = torch.from_numpy(np.array(audio[random_idx:random_idx+self.n_samples]).astype('float32'))

        audio_tensor = audio.to(torch.float32)

        return audio_tensor

    def __getitem__(self, index):
        return {
            "audio": self.load_audio(index),
            "text_input": self.mtt_dataset[index]["gt_caption"],
            "instruction_input": self.instruction
        }
    
    def collater(self, samples):
        audios = [s['audio'] for s in samples]
        audio_sizes = [len(s['audio']) for s in samples]
        audio_size = max(audio_sizes)
        txts = [s['text_input'] for s in samples]

        instructions = [s['instruction_input'] for s in samples]

        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        attn_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(True)
        )

        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            else: #diff < 0
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                attn_mask[i, diff:] = False

        attn_mask = attn_mask.int()

        return {'audio': collated_audios, 'text_input': txts, 'instruction_input': instructions, 'attention_mask': attn_mask}

    def __len__(self):
        return len(self.mtt_dataset)