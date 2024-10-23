"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from musilingo.common.registry import registry
from musilingo.tasks.base_task import BaseTask
from musilingo.datasets.data_utils import prepare_sample
from musilingo.common.logger import MetricLogger
from tqdm import tqdm
import torch


@registry.register_task("audio_text_pretrain")
class AudioTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, output_dir, cur_epoch, split_name, cuda_enabled=True):
        model.eval()

        preds = []
        gts = []

        for _ in tqdm(range(len(data_loader))):
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            with torch.cuda.amp.autocast(enabled=True):
                # predictions, gt = self.valid_step(model=model, samples=samples)
                predictions, gt = model.evaluate_step(samples)

            predictions = [pred.replace("\n", " ") for pred in predictions]

            preds.extend(predictions)
            gts.extend(gt)

        with open(f"{output_dir}/predictions_{cur_epoch}_{split_name}.txt", "w") as f:
            f.write("\n".join(preds))
        
        with open(f"{output_dir}/ground_truth_{cur_epoch}_{split_name}.txt", "w") as f:
            f.write("\n".join(gts))