import os
import logging
import warnings
import torch
from transformers import Wav2Vec2FeatureExtractor

from musilingo.common.registry import registry
from musilingo.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from musilingo.datasets.datasets.musiccaps_dataset import MusicCapsDataset
from musilingo.datasets.datasets.musicqa_dataset import MusicQADataset
from musilingo.datasets.datasets.msd_dataset import MSDDataset
from musilingo.datasets.datasets.cmi_dataset import CMIDataset
from musilingo.datasets.datasets.mtt_dataset import MTTDataset


@registry.register_builder("musiccaps")
class MusicCapsBuilder(BaseDatasetBuilder):
    train_dataset_cls = MusicCapsDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/musiccaps/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        #self.build_processors()

        #build_info = self.config.build_info
        #storage_path = build_info.storage

        datasets = dict()

        #if not os.path.exists(storage_path):
        #    warnings.warn("storage path {} does not exist.".format(storage_path))

        #get processor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(self.config.processor, trust_remote_code=True)


        # create datasets
        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            processor=processor,
            split=self.config.split, 
            data_dir=self.config.data_dir, 
        )
 

        return datasets

@registry.register_builder("cmi")
class CMIBuilder(BaseDatasetBuilder):
    train_dataset_cls = CMIDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cmi/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        #self.build_processors()

        #build_info = self.config.build_info
        #storage_path = build_info.storage

        datasets = dict()

        #if not os.path.exists(storage_path):
        #    warnings.warn("storage path {} does not exist.".format(storage_path))

        #get processor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(self.config.processor, trust_remote_code=True)


        # create datasets
        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            processor=processor,
            split=self.config.split, 
            data_dir=self.config.data_dir, 
            question_type=self.config.question_type
        )
 

        return datasets

@registry.register_builder("musicqa")
class MusicQABuilder(BaseDatasetBuilder):
    train_dataset_cls = MusicQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/musicqa/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        #self.build_processors()

        #build_info = self.config.build_info
        #storage_path = build_info.storage

        datasets = dict()

        #if not os.path.exists(storage_path):
        #    warnings.warn("storage path {} does not exist.".format(storage_path))

        #get processor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(self.config.processor, trust_remote_code=True)


        # create datasets
        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            processor=processor,
            split=self.config.split, 
            data_dir=self.config.data_dir, 
        )
 

        return datasets

@registry.register_builder("msd")
class MSDBuilder(BaseDatasetBuilder):
    train_dataset_cls = MSDDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msd/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        #self.build_processors()

        #build_info = self.config.build_info
        #storage_path = build_info.storage

        datasets = dict()

        #if not os.path.exists(storage_path):
        #    warnings.warn("storage path {} does not exist.".format(storage_path))

        #get processor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(self.config.processor, trust_remote_code=True)


        # create datasets
        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            processor=processor,
            split=self.config.split,
            mp3_file_folder=self.config.data_dir,
        )


        return datasets


@registry.register_builder("mtt")
class MTTBuilder(BaseDatasetBuilder):
    train_dataset_cls = MTTDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mtt/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        #self.build_processors()

        #build_info = self.config.build_info
        #storage_path = build_info.storage

        datasets = dict()

        #if not os.path.exists(storage_path):
        #    warnings.warn("storage path {} does not exist.".format(storage_path))

        #get processor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(self.config.processor, trust_remote_code=True)


        # create datasets
        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            processor=processor,
            split=self.config.split,
            genre=self.config.genre,
            pct_other_genres=self.config.pct_other_genres,
            top_k=self.config.top_k,
            tf_idf=self.config.tf_idf,
            unit_type=self.config.unit_type,
            arch=self.config.arch
        )

        datasets['valid'] = dataset_cls(
            processor=processor,
            split="valid",
            genre=self.config.genre,
            pct_other_genres=self.config.pct_other_genres,
            top_k=self.config.top_k,
            tf_idf=self.config.tf_idf,
            unit_type=self.config.unit_type,
            arch=self.config.arch
        )

        datasets['test'] = dataset_cls(
            processor=processor,
            split="test",
            genre=self.config.genre,
            pct_other_genres=self.config.pct_other_genres,
            top_k=self.config.top_k,
            tf_idf=self.config.tf_idf,
            unit_type=self.config.unit_type,
            arch=self.config.arch
        )


        return datasets