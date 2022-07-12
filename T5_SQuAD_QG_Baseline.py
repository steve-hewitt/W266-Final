# Import block
import json
import argparse
import time
import os
import pandas as pd

from datasets import Dataset
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
#import tensorflow_datasets as tfds

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from packaging import version

from datasets import list_datasets, load_dataset, list_metrics, load_metric
from datasets import Dataset

# Utility functions from GP-VAE implementation

#def parse_data(in_file='../../data/GYAFC/em/trn.tsv'):
#    with open(in_file, 'r') as f:
#        data = f.read().split('\n')
#        data.remove('')
#    contexted = []
#    for i, line in enumerate(data):
#        source_txt = line.split('\t')[0]
#        target_txt = line.split('\t')[1]
#        row = (i, source_txt, target_txt)
#        contexted.append(row)
#    columns = ['id', 'source', 'target']
#    data_df = pd.DataFrame.from_records(contexted, columns=columns)
#    return data_df

# Specific to dataset.
def construct_input_for_batch(tokenizer, batch, args):
    """
    Function that takes a batch from a dataset and constructs the corresponding 
    input string.
    """
    source, target = [], []
    for inp, out in zip(batch['source'], batch['target']):
        source.append(inp.strip())
        target.append(out.strip())
    if batch['id'][0] == 0:
        print(source[0])
        print(target[0])
        print()
    return source, target

def make_batch_inputs(batch, tokenizer, args, device='cuda:0'):
  """
  Function that takes a batch from a dataset and transforms it 
  """
  # Concatenate the concept names for each example in the batch.
  input_lists, _ = construct_input_for_batch(tokenizer, batch, args)
  # Use the model's tokenizer to create the batch input_ids.
  batch_features = tokenizer(input_lists, padding=True, return_tensors='pt')
  # Move all inputs to the device.
  batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])
  return batch_features

def make_batch_data(batch, tokenizer, args, device='cuda:0'):
  """
  Function that takes a batch from a dataset and transforms it 
  """
  # Concatenate the concept names for each example in the batch.
  input_lists, label_list = construct_input_for_batch(tokenizer, batch, args)
  # Use the model's tokenizer to create the batch input_ids.
  batch_features = tokenizer(input_lists, padding=True, return_tensors='pt')
  batch_labels = tokenizer(label_list, padding=True, return_tensors='pt')
  # Move all inputs to the device.
  batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])
  batch_labels = dict([(k, v.to(device)) for k, v in batch_labels.items()])
  return batch_features, batch_labels

def batch_tokenize(dataset_batch, tokenizer, args):
  """
  Reuse the function defined above to construct the batch (source, target) and 
  run them through the tokenizer.
  """
  source, target = construct_input_for_batch(tokenizer, dataset_batch, args)
  res = {
          "input_ids": tokenizer(
              source,
              padding='max_length', 
              truncation=True,
              max_length=args.encoder_max_length
          )["input_ids"],
          "labels": tokenizer(
              target,
              padding='max_length', 
              truncation=True,
              max_length=args.decoder_max_length
          )["input_ids"],
  }
  return res

def batchify_data(df, tokenizer, args):
  dataset = Dataset.from_pandas(df)
  data_tokenized = dataset.map(
    lambda batch: batch_tokenize(batch, tokenizer, args),
    batched=True
  )
  return data_tokenized

def compute_loss(batch, model, tokenizer, args):
  batch_feature, batch_label = make_batch_data(batch, tokenizer, args)
  with torch.no_grad():
    outputs = model(input_ids=batch_feature['input_ids'],
                    labels=batch_label['input_ids'])
    eval_loss = outputs.loss.item()
  return [eval_loss] 

def test_ppl(val_df, model, tokenizer, args):
  loss_dict = Dataset.from_pandas(val_df).map(
    lambda batch: {'loss': compute_loss(batch, model, tokenizer, args)},
    batched=True,
    batch_size=1,
  )
  
  eval_loss = 0.
  nb_eval_steps = 0
  for item in list(loss_dict):
      eval_loss += item['loss']
      nb_eval_steps += 1
  eval_loss = eval_loss / nb_eval_steps
  ppl = torch.exp(torch.tensor(eval_loss))
  return ppl.item()

def prepare_eval(output_list):
    ref_list, pred_list = [], []
    for item in output_list:
        pred_list.append({"generated": item['generated']})
        ref_list.append({"target": [item['target']]})
    return ref_list, pred_list

# Replacing dataset constructing function from utilities with a custom one.
def parse_data(t_split='train'):
  # Get dataset.
  #squad_dataset = load_dataset('squad') # Method A - standard version
  #squad_dataset, info = tfds.load('squad_question_generation/split_zhou', with_info=True, split='train') # Method B - version with split that could be better.

  # Split handling - validation set further split into 50% dev/test.
  if t_split == 'train':
    df = pd.DataFrame(load_dataset('squad')['train'])
  elif t_split in ['val','test']:
    vt_df = pd.DataFrame(load_dataset('squad')['validation'])
    df_val = vt_df.sample(frac=0.5,random_state=266)
    if t_split == 'test':
      df_test = vt_df.drop(df_val.index)
      df = df_test
    else:
      df = df_val
  else:
    raise Exception("Invalid choice of dataset split.")
  

  df['answer_text'] = df['answers'].apply(lambda x: x['text'][0])
  df['source'] = 'answer: ' + df['answer_text'] + ' context: ' + df['context'] + '</s>'
  df['target'] = df['question']

  return df                                                                                                                       

if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

class Seq2SeqTrainer(Trainer):
    """Class to finetune a Seq2Seq model."""

    def __init__(
            self,
            num_beams=4,
            max_length=32,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_beams = num_beams
        self.max_length = max_length

    def compute_loss(self, model, inputs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        outputs = model(input_ids=inputs['input_ids'],
                        # decoder_input_ids=inputs['labels'][:,:-1],
                        labels=inputs['labels'])
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.label_smoother is not None and "labels" in inputs:
            return self.label_smoother(outputs, inputs["labels"])
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            return outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Runs the model to either generate a sequence and/or compute the loss.
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        # Compute loss with labels first.
        with torch.no_grad():
            if self.args.fp16 and _use_native_amp:
                with autocast():
                    outputs = model(input_ids=inputs['input_ids'],
                                    # decoder_input_ids=inputs['labels'][:,:-1],
                                    labels=inputs['labels'])
            else:
                outputs = model(input_ids=inputs['input_ids'],
                                # decoder_input_ids=inputs['labels'][:,:-1],
                                labels=inputs['labels'])
            if has_labels:
                loss = outputs[0].mean().detach()
            else:
                loss = None
        # If we're only computing the conditional log-likelihood, return.
        if prediction_loss_only:
            return (loss, None, None)
        # Otherwise run model.generate() to get predictions.
        if isinstance(model, torch.nn.DataParallel):
            preds = model.module.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                num_beams=self.num_beams,
                max_length=self.max_length,
            )
        else:
            preds = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                num_beams=self.num_beams,
                max_length=self.max_length,
            )
        if len(preds) == 1:
            preds = preds[0]
        # Pad predictions if necessary so they can be concatenated across batches.
        if preds.shape[-1] < self.max_length:
            preds = torch.nn.functional.pad(
                preds, (0, self.max_length - preds.shape[-1]),
                mode='constant',
                value=self.tokenizer.pad_token_id
            )
        # Post-process labels.
        if has_labels:
            labels = inputs.get('labels')
        else:
            labels = None
        return (loss, preds, labels)


def train(args):
    # Load the dataset
    #trn_df = parse_data(in_file=f'../../data/{args.dataset}/trn.tsv')
    #val_df = parse_data(in_file=f'../../data/{args.dataset}/val.tsv')
    trn_df = parse_data('train')
    val_df = parse_data('val')

    # Load the pre-trained model
    ckpt_path = None
    if args.task == 'train':
        ckpt_path = args.model_name
    else:
        ckpt_path = f"{args.model_name}_{args.dataset}_{args.flag}_{args.kernel_v}_{args.kernel_r}_{args.timestamp}/checkpoint-{args.ckpt}"
        # update timestamp and create new path for ckpt
        args.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    tokenizer = T5TokenizerFast.from_pretrained(ckpt_path)
    print(f"Vocab size: {len(tokenizer)}")

    train_data_tokenized = batchify_data(trn_df, tokenizer, args)
    valid_data_tokenized = batchify_data(val_df, tokenizer, args)

    model = T5ForConditionalGeneration.from_pretrained(ckpt_path)
    model = model.to('cuda:0')

    # Training Setup
    train_args = TrainingArguments(
        output_dir=f"{args.model_name}_{args.dataset}_{args.flag}_{args.kernel_v}_{args.kernel_r}_{args.timestamp}",
        do_train=True,
        do_eval=True,
        save_strategy="steps",
        save_steps=300,
        evaluation_strategy="steps",
        eval_steps=300,
        logging_steps=100,
        # optimization args, the trainer uses the Adam optimizer
        # and has a linear warmup for the learning rate
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1e-04,
        num_train_epochs=args.epochs,
        warmup_steps=0,
        lr_scheduler_type='constant',
        # misc args
        seed=42,
        save_total_limit=10,  # limit the total amount of checkpoints
        disable_tqdm=False,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        local_rank=args.local_rank
    )

    trainer = Seq2SeqTrainer(
        num_beams=args.beam_size,
        max_length=args.decoder_max_length,
        model=model,
        args=train_args,
        train_dataset=train_data_tokenized,
        eval_dataset=valid_data_tokenized,
        tokenizer=tokenizer,
    )

    # Now that we have the trainer set up, we can finetune.
    trainer.train()


def beam_generate_sentences(batch,
                            model,
                            tokenizer,
                            args,
                            device='cuda:0'):
    # Create batch inputs.
    features = make_batch_inputs(
        batch=batch,
        tokenizer=tokenizer,
        args=args,
        device=device)
    # Generate with beam search.
    generated_ids = model.generate(
        input_ids=features['input_ids'],
        attention_mask=features['attention_mask'],
        num_beams=args.beam_size,
        max_length=args.max_generation_length,
        num_return_sequences=1,
    )
    # Use model tokenizer to decode to text.
    generated_sentences = [
        tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
        for gen_ids in generated_ids
    ]
    print(generated_sentences)
    return ['\t'.join(generated_sentences)]


def sample_sentences(batch,
                     model,
                     tokenizer,
                     args,
                     device='cuda:0'):
    # Create batch inputs.
    features = make_batch_inputs(
        batch=batch,
        tokenizer=tokenizer,
        args=args,
        device=device)

    generated_sentences = []
    for i in range(args.num_return_sequences):
        # Generate with beam search.
        generated_ids = model.generate(
            input_ids=features['input_ids'],
            attention_mask=features['attention_mask'],
            num_beams=args.beam_size,
            max_length=args.max_generation_length,
            num_return_sequences=1,
        )
        # Use model tokenizer to decode to text.
        generated_sentences += [
            tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
            for gen_ids in generated_ids
        ]
    print(generated_sentences)
    return ['\t'.join(generated_sentences)]


def test(args):
    te_df = parse_data('test')
    print('Data loaded!!!')

    # Load the model
    if args.timestamp == '0':
        tokenizer = T5TokenizerFast.from_pretrained(f"{args.model_name}")
    else:
        ckpt_path = f"{args.model_name}_{args.dataset}_{args.flag}_{args.kernel_v}_{args.kernel_r}_{args.timestamp}/checkpoint-{args.ckpt}"
        tokenizer = T5TokenizerFast.from_pretrained(ckpt_path)
    print(f"Vocab size: {len(tokenizer)}")

    if args.timestamp == '0':
        model = T5ForConditionalGeneration.from_pretrained(f"{args.model_name}")
    else:
        ckpt_path = f"{args.model_name}_{args.dataset}_{args.flag}_{args.kernel_v}_{args.kernel_r}_{args.timestamp}/checkpoint-{args.ckpt}"
        model = T5ForConditionalGeneration.from_pretrained(ckpt_path)
    model = model.to('cuda:0')
    model.kernel_v = args.kernel_v
    model.kernel_r = args.kernel_r
    model.from_mean = args.from_mean
    model.scaler = args.scaler

    # Make predictions
    if args.from_mean:
        test_output = Dataset.from_pandas(te_df).map(
            lambda batch: {'generated': beam_generate_sentences(
                batch,
                model,
                tokenizer,
                args,
                device='cuda:0')
            },
            batched=True,
            batch_size=1,
        )
    else:
        test_output = Dataset.from_pandas(te_df).map(
            lambda batch: {'generated': sample_sentences(
                batch,
                model,
                tokenizer,
                args,
                device='cuda:0')
            },
            batched=True,
            batch_size=1,
        )

    # prepare evaluation data
    ref_list, pred_list = prepare_eval(list(test_output))
    reference_dict = {
        "language": "en",
        "values": ref_list,
    }
    prediction_dict = {
        "language": "en",
        "values": pred_list,
    }

    if args.timestamp == '0':
        os.makedirs(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}")

    with open(
            f"{args.model_name}_{args.dataset}_{args.flag}_{args.kernel_v}_{args.kernel_r}_{args.timestamp}/refs.json",
            'w') as f:
        f.write(json.dumps(reference_dict, indent=2))
    if args.from_mean:
        with open(
                f"{args.model_name}_{args.dataset}_{args.flag}_{args.kernel_v}_{args.kernel_r}_{args.timestamp}/outs_mean.json",
                'w') as f:
            f.write(json.dumps(prediction_dict, indent=2))
    else:
        with open(
                f"{args.model_name}_{args.dataset}_{args.flag}_{args.kernel_v}_{args.kernel_r}_{args.timestamp}/outs.json",
                'w') as f:
            f.write(json.dumps(prediction_dict, indent=2))
            
if __name__ == '__main__':

    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-t', '--task', type=str, default="train",
                    help="specify the task to do: (train)ing, ft(finetune), (eval)uation")
    #p.add_argument('-t', '--task', type=str, default="eval",
    #                help="specify the task to do: (train)ing, ft(finetune), (eval)uation")
    #p.add_argument('-t', '--task', type=str, default="ft",
    #                help="specify the task to do: (train)ing, ft(finetune), (eval)uation")
    #p.add_argument('-c', '--ckpt', type=str, default="193280",
    #                help="Model checkpoint")
    #p.add_argument('-time', '--timestamp', type=str, default='2021-02-14-04-57-04',
    #                help="Model checkpoint")
    p.add_argument('-c', '--ckpt', type=str, default="30000",
                    help="Model checkpoint")
    p.add_argument('-time', '--timestamp', type=str, default='2022-07-10-18-08-18',
                    help="Model checkpoint")
    p.add_argument('-f', '--flag', type=str, default='gpvae',
                    help="Model checkpoint")
    p.add_argument('-d', '--dataset', type=str, default="GYAFC/em",
                    help="specify the dataset: GYAFC/em, GYAFC/fr")
    p.add_argument('--model_name', type=str, default="t5-base",
                    help="specify the model name: t5-base, facebook/blenderbot-400M-distill")
    p.add_argument('-v', '--kernel_v', type=float, default=64.0,
                    help="Hyper-parameter for prior kernel,  control the signal variance")
    p.add_argument('-r', '--kernel_r', type=float, default=0.0001,
                    help="Hyper-parameter for prior kernel.")
    p.add_argument('-s', '--scaler', type=float, default=1.0)
    p.add_argument('--from_mean', action='store_true',
                    help="specify whether sample from mean during generation")
    p.add_argument('-bz', '--batch_size', type=int, default=16)
    #p.add_argument('-bz', '--batch_size', type=int, default=4)
    p.add_argument('-e', '--epochs', type=int, default=10)
    #p.add_argument('--encoder_max_length', type=int, default=50)
    #p.add_argument('--encoder_max_length', type=int, default=512)
    p.add_argument('--encoder_max_length', type=int, default=128)
    p.add_argument('--decoder_max_length', type=int, default=48)
    #p.add_argument('--max_generation_length', type=int, default=60)
    p.add_argument('--max_generation_length', type=int, default=96)
    #p.add_argument('--beam_size', type=int, default=10)
    p.add_argument('--beam_size', type=int, default=5)
    #p.add_argument('--num_return_sequences', type=int, default=10)
    p.add_argument('--num_return_sequences', type=int, default=5)
    p.add_argument('--local_rank', type=int, default=-1,
                    help="Multiple GPU training")
    args = p.parse_args()

    # jupyter fix for bad flag
    args.flag = 'gpvae'

    if args.task == 'train':
        args.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        train(args)
    elif args.task == 'ft':
        train(args)
    else:
        test(args)                                                                                                        