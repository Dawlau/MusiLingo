from collections import Counter
import json
import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from musilingo.common.registry import registry
from musilingo.models.llama_model import LlamaForCausalLM
from musilingo.models.base_model import BaseModel
from transformers import LlamaTokenizer, Wav2Vec2FeatureExtractor, AutoModel
from transformers import StoppingCriteria, StoppingCriteriaList

import torchaudio.transforms as T


@registry.register_model("musilingo")
class MusiLingo(BaseModel):
    """
    MERT GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/musilingo.yaml",
    }

    def __init__(
        self,
        mert_model="m-a-p/MERT-v1-330M",
        llama_model="lmsys/vicuna-7b-v1.5",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        audio_first=True,
        cer=False,
        dataset_config=None,
        audio_interpolation=False
    ):
        super().__init__()

        self.low_resource = low_resource

        self.audio_first = audio_first
        self.cer = cer
        self.dataset_config = dataset_config
        self.audio_interpolation = audio_interpolation

        self.genre = self.dataset_config.get("genre", "")
        self.top_k = self.dataset_config.get("top_k", 0)
        self.tf_idf = self.dataset_config.get("tf_idf", False)
        self.unit_type = self.dataset_config.get("unit_type", "unigram")

        print('Loading Audio Encoder')
        self.audio_encoder = AutoModel.from_pretrained(mert_model, trust_remote_code=True)
        # loading the corresponding preprocessor config
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_model, trust_remote_code=True)

        for name, param in self.audio_encoder.named_parameters():
            param.requires_grad = False
        self.audio_encoder = self.audio_encoder.eval()

        print('Loading Audio Encoder Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit},
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )
        # self.llama_model = self.llama_model.to("cuda")
        # inputs = self.llama_tokenizer("###Human: Tell me a story ###Assistant: ", return_tensors="pt", padding=True, truncation=True).to(self.llama_model.device)
    
        # # Generate response from the model
        # with torch.cuda.amp.autocast(enabled=True):
            # outputs = self.llama_model.generate(
        #         input_ids=inputs['input_ids'],
        #         attention_mask=inputs['attention_mask'],
        #         max_length=300,
        #         do_sample=True,  # Enable sampling for diverse responses
        #         top_k=50,  # Top-k sampling for more interesting output
        #         top_p=0.95,  # Nucleus sampling
        #         temperature=0.7  # Adjust temperature for creativity
        #     )

        # # Decode the generated response
        # answer = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(answer)
        # exit(0)

        self.llama_model.class_weights = None
        if self.cer and self.genre != "all":
            with open(f"genres_vocab/{self.genre}_vocab_{self.unit_type}_tf_idf_{self.tf_idf}.json", "r") as f:
                counts_vocabulary = json.load(f)

            vocabulary = [(word, freq) for word, freq in Counter(counts_vocabulary).most_common(None if self.top_k == -1 else self.top_k)]

            token_ids = dict()
            total_freq = 0

            for word, freq in vocabulary:
                tokens = self.llama_tokenizer(word, add_special_tokens=False, return_tensors="pt")["input_ids"].tolist()[0]
                for token in tokens:
                    if token not in token_ids:
                        token_ids[token] = 0
                    token_ids[token] += freq
                    total_freq += freq


            N = self.llama_tokenizer.vocab_size

            weights = torch.empty(N).fill_((1 - self.cer) / (N - len(token_ids)))

            for token, freq in token_ids.items():
                weights[token] = token_ids[token] * self.cer / total_freq

            weights = weights * N

            self.llama_model.class_weights = weights

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.audio_encoder.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        self.prompt_template = prompt_template

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<AudioHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def audioenc_to_cpu(self):
        self.audio_encoder.to("cpu")
        self.audio_encoder.float()

    def encode_audio(self, audio, attn=None):
        device = audio.device
        if self.low_resource:
            self.audioenc_to_cpu()
            audio = audio.to("cpu")

        if attn is None:

            audio_embeds = torch.stack(self.audio_encoder(input_values=audio, 
                                                          output_hidden_states=True).hidden_states) # [25, B, T, 1024]
            audio_embeds = audio_embeds.transpose(0, 1).mean(-3) #[B, T, 1024]

        else:
  
            audio_embeds = torch.stack(self.audio_encoder(input_values=audio, 
                                                          output_hidden_states=True, 
                                                          attention_mask=attn).hidden_states) # [25, B, T, 1024]
            audio_embeds = audio_embeds.transpose(0, 1).mean(-3) #[B, T, 1024]
            
        # Average time steps:
        t = 325
        B, T, D = audio_embeds.shape
        avg_tmp = audio_embeds[:, :T//t*t].reshape(B, T//t, t, D).mean(2)

        # Average the remaining steps
        if T % t > 0:
          avg_last = audio_embeds[:, T//t*t:].reshape(B, 1, T%t, D).mean(2)
          audio_embeds = torch.concat([avg_tmp, avg_last], dim=1)
        else:
          audio_embeds = avg_tmp
        audio_embeds = audio_embeds.to(device)
        inputs_llama = self.llama_proj(audio_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(audio.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, audio_embeds, atts_audio, prompt):
        if prompt:
            batch_size = audio_embeds.shape[0]
            p_before, p_after = prompt.split('<AudioHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(audio_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(audio_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_audio_embeds = torch.cat([p_before_embeds, audio_embeds, p_after_embeds], dim=1)
            wrapped_atts_audio = atts_audio[:, :1].expand(-1, wrapped_audio_embeds.shape[1])
            return wrapped_audio_embeds, wrapped_atts_audio
        else:
            return audio_embeds, atts_audio
        
    def instruction_prompt_wrap(self, audio_embeds, atts_audio, prompt):
        if prompt:
            batch_size = audio_embeds.shape[0]
            p_before = []
            p_after = []
            p = []

            for i in range(batch_size):
                p_b, p_a = prompt[i].split('<AudioHere>')
                p_before.append(p_b)
                p_after.append(p_a)
            #     p.append(p_b + p_a)
            #     p[i] = p[i].replace("<Audio></Audio>", "")

            # print(p)

            # p_tokens = self.llama_tokenizer(
            #     p, return_tensors="pt", add_special_tokens=False, padding="longest").to(audio_embeds.device)
            
            # p_embeds = self.llama_model.model.embed_tokens(p_tokens.input_ids)
            # p_atts = p_tokens.attention_mask

            # return p_embeds, p_atts
  
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", padding='longest', add_special_tokens=True).to(audio_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", padding='longest', add_special_tokens=False).to(audio_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
            wrapped_audio_embeds = torch.cat([p_before_embeds, audio_embeds, p_after_embeds], dim=1)
            wrapped_audio_embeds = torch.cat([p_before_embeds, p_after_embeds], dim=1)
            wrapped_atts_audio = torch.cat([p_before_tokens.attention_mask, atts_audio, p_after_tokens.attention_mask], dim=1)
            wrapped_atts_audio = torch.cat([p_before_tokens.attention_mask, p_after_tokens.attention_mask], dim=1)
            return wrapped_audio_embeds, wrapped_atts_audio
        else:
            return audio_embeds, atts_audio


    def forward(self, samples):
        audio = samples["audio"]
        attn = samples["attention_mask"] if "attention_mask" in samples else None
        audio_embeds, atts_audio = self.encode_audio(audio, attn)

        if 'instruction_input' in samples:  # instruction tuning dataset
            instruction_prompt = []
            for instruction in samples['instruction_input']:
                if self.audio_first:
                    prompt = '<Audio><AudioHere></Audio> ' + instruction
                else:
                    prompt = instruction + ' <Audio><AudioHere></Audio>'
                instruction_prompt.append(self.prompt_template.format(prompt))

            audio_embeds, atts_audio = self.instruction_prompt_wrap(audio_embeds, atts_audio, instruction_prompt)

        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            audio_embeds, atts_audio = self.prompt_wrap(audio_embeds, atts_audio, prompt)

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(audio.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_audio.shape[0], atts_audio.shape[1]+1],
                       dtype=torch.long).to(audio.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = audio_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_audio[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, audio_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_audio, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss, "logits": outputs.logits}
    
    def evaluate_step(self, samples):
        class StoppingCriteriaSub(StoppingCriteria):
            def __init__(self, stops=[], encounters=1):
                super().__init__()
                self.stops = stops
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                for stop in self.stops:
                    if torch.all((stop == input_ids[0][-len(stop):])).item():
                        return True
                return False

        stopping = StoppingCriteriaList([StoppingCriteriaSub([torch.tensor([835]).cuda(),
                                    torch.tensor([2277, 29937]).cuda()])])

        audio = samples["audio"].cuda()
        audio_embeds, atts_audio = self.encode_audio(audio)

        if 'instruction_input' in samples:  # instruction dataset
            #print('Instruction Batch')
            instruction_prompt = []
            for instruction in samples['instruction_input']:
                if self.audio_first:
                    prompt = '<Audio><AudioHere></Audio> ' + instruction
                else:
                    prompt = instruction + ' <Audio><AudioHere></Audio>'
                instruction_prompt.append(self.prompt_template.format(prompt))

            audio_embeds, atts_audio = self.instruction_prompt_wrap(audio_embeds, atts_audio, instruction_prompt)

        self.llama_tokenizer.padding_side = "right"
        batch_size = audio_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                        dtype=torch.long,
                        device=torch.device('cuda')) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_audio[:, :1]
        inputs_embeds = torch.cat([bos_embeds, audio_embeds], dim=1)
        inputs_atts = torch.cat([atts_bos, atts_audio], dim=1)
        
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs_atts,
            max_new_tokens=300,
            stopping_criteria=stopping,
            num_beams=1,
            do_sample=True,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=0.1,
        )

        # for output in outputs:
        #     output_token = output
        #     # if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
        #     #     output_token = output_token[1:]
        #     # if output_token[0] == 1:  # if there is a start token <s> at the beginning. remove it
        #     #     output_token = output_token[1:]
        #     output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        #     output_text = output_text.split('###')[0]  # remove the stop sign '###'
        #     output_text = output_text.split('Assistant:')[-1].strip()

        # for output in outputs:
            # output_token = output
            # if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            #     output_token = output_token[1:]
            # if output_token[0] == 1:  # if there is a start token <s> at the beginning. remove it
            #     output_token = output_token[1:]
        text = outputs[0]
        if text[0] == 0:
            text = text[1:]
        if text[0] == 1:
            text = text[1:]
        output_text = self.llama_tokenizer.decode(text, add_special_tokens=False)
            # output_text = output_text.split('###')[0]  # remove the stop sign '###'
            # output_text = output_text.split('Assistant:')[-1].strip()
        print("Output: ", output_text)

        text = outputs[1]
        if text[0] == 0:
            text = text[1:]
        if text[0] == 1:
            text = text[1:]
        output_text = self.llama_tokenizer.decode(text, add_special_tokens=False)
            # output_text = output_text.split('###')[0]  # remove the stop sign '###'
            # output_text = output_text.split('Assistant:')[-1].strip()
        print("Output: ", output_text)

        # exit(0)

        return output_text


    @classmethod
    def from_config(cls, cfg):
        mert_model = cfg.get("mert_model", "")
        llama_model = cfg.get("llama_model")

        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        audio_first = cfg.get("audio_first", True)
        audio_interpolation = cfg.get("audio_interpolation", 0.5)
        cer = cfg.get("cer", False)
        dataset_config = cfg.get("mtt", None)

        # genre = cfg.get("genre", "")
        # top_k = cfg.get("top_k", 0)
        # tf_idf = cfg.get("tf_idf", 0)
        # unit_type = cfg.get("unit_type", "unigram")

        model = cls(
            mert_model=mert_model,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            audio_first=audio_first,
            cer=cer,
            dataset_config=dataset_config,
            audio_interpolation=audio_interpolation,
        )

        ckpt_path = cfg.get("ckpt", "")  # load ckpt weights of MusiLingo
        if ckpt_path:
            print("Load MERT-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

            # print(ckpt["model"])

            # for x in ckpt["model"]:
            #     print(x)

            # exit(0)

        return model
