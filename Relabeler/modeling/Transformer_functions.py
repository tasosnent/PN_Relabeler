import os.path

import torch
from datetime import datetime
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import *
import time
from sklearn.metrics import f1_score, classification_report
import wandb
import copy
from BertModeling import BertForMultiLabelSequenceClassification
from other_functions import *

# Basic functions for model development and evaluation

def format_for_bert(label_list, sentences_list, tokenizer, verbose = True):
    '''
        Tokenize all the sentences and labels-names in each (labels-name, sentence) pair and map the tokens to their word IDs.
    :param label_list:      A list of label-names, parallel to sentences_list, to be tokenized as well.
    :param sentences_list:  A list with the sentences (article titles+abstracts) to tokenize
    :param tokenizer:       The toeknizer to be used
    :param verbose:         whether to print details
    :return:                Two tensors (input_ids, attention_masks) with tokenized text ([CLS]label-name[SEP]title+abstract[SEP]) and corresponding mask.
    '''
    input_ids = []
    attention_masks = []

    # For every sentence...
    for lab, sent in zip(label_list, sentences_list):
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            lab,  # Sentence to encode.
            text_pair = sent,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            # max_length = 256,           # Pad & truncate all sentences.
            max_length=512,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        if verbose and len(input_ids)%10000 == 0:
            ratio = round((len(input_ids)/len(sentences_list))*100,2)
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  len(input_ids), " sentences processed (",ratio,"%)")
    if verbose and len(input_ids)%10000 == 0:
        ratio = round((len(input_ids)/len(sentences_list))*100,2)
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              len(input_ids), " sentences processed (",ratio,"%)")
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks
    # labels = yy

def get_sentences(dataset):
    '''
        Read a dataframe dataset and get the labels ('label'), texts ('text'), and y values ('target').
    :param dataset:     The dataframe with columns 'label', 'text', 'target'
    :return:            label_list (List of stings), sentences_list (List of stings), yy (tensor)
    '''
    sentences_list = []
    label_list = []
    yy = []
    for i, row in dataset.iterrows():
      # print(row)
      dx = row['text']
      dx2 = row['label']
      y = []
      y.append(row['target'])
      # print(y)
      yy.append(y)

      sentences_list.append(dx)
      label_list.append(dx2)
    yy = torch.Tensor(yy)
    yy = yy.double()
    return label_list, sentences_list, yy

def read_training_data(data_folder, target_labels, target_label_names, balance_n, seed):
    '''
        Read and fuse the datasets to be used for development
    :param data_folder:          Path to folder where the filtering datasets are stored
    :param target_labels:        A list of labels to consider. (e.g. ['D049920'] )
    :param target_label_names:   A list of labels names parallel to target_labels. (e.g. ['Ralstonia pickettii'] )
    :param balance_n:            A number indicating the desired negative-to-positive-instances ratio.
    :param seed:                 A seed value for reproducibility of "random" choices
    :return:                     Two dataframes train_data, and val_data
    '''

    print('\tStart read_training_data :', target_label_names)
    input_train_data = None
    input_val_data = None
    for target_label, target_label_name in zip(target_labels, target_label_names):
        print('\tWork on :',  target_label, " > ", target_label_name)
        training_csv = data_folder + os.path.sep + 'f2005_full_' + target_label + os.path.sep + 'train.csv'
        val_csv = data_folder + os.path.sep + 'f2005_full_' + target_label + os.path.sep + 'dev.csv'
        train_data_tmp = read_data(training_csv, target_label, target_label_name )
        val_data_tmp = read_data(val_csv, target_label, target_label_name )
        print('\t\tOriginal train : size = ',  str(len(train_data_tmp)), ", positive = ", str(train_data_tmp['target'].sum()))
        train_data_tmp = balance_dataset(train_data_tmp, 'target', balance_n, seed)
        print('\t\tBalanced train : size = ',  str(len(train_data_tmp)), ", positive = ", str(train_data_tmp['target'].sum()))
        print('\t\tOriginal val : size = ',  str(len(val_data_tmp)), ", positive = ", str(val_data_tmp['target'].sum()))
        val_data_tmp = balance_dataset(val_data_tmp, 'target', balance_n, seed)
        print('\t\tBalanced val : size = ',  str(len(val_data_tmp)), ", positive = ", str(val_data_tmp['target'].sum()))

        if input_train_data is None:
            input_train_data = train_data_tmp
            input_val_data = val_data_tmp
        else:
            input_train_data = input_train_data.append(train_data_tmp)
            input_val_data = input_val_data.append(val_data_tmp)
        print('\ttotal train data :', input_train_data.shape, " adding ", target_label, " > ", target_label_name)

    print('\ttotal train data:', input_train_data.shape)
    positive_instances = input_train_data['target'].sum()
    print('\t\t poisitve:', positive_instances)

    print('\ttotal val data:', input_val_data.shape)
    positive_instances = input_val_data['target'].sum()
    print('\t\t poisitve:', positive_instances)

    train_data = input_train_data.sample(frac=1, random_state=seed)
    val_data = input_val_data.sample(frac=1, random_state=seed)

    return train_data, val_data

def read_test_data(data_folder, target_labels, target_label_names, seed = 10):
    '''
            Read and fuse the datasets to be used for testing
    :param data_folder:          Path to folder where the testing datasets are stored
    :param target_labels:        A list of labels to consider. (e.g. ['D049920'] )
    :param target_label_names:   A list of labels names parallel to target_labels. (e.g. ['Ralstonia pickettii'] )
    :param seed:                 A seed value for reproducibility of "random" choices
    :return:                     A dataframes test_data
    '''
    input_train_data = None
    for target_label, target_label_name in zip(target_labels, target_label_names):
        training_csv = data_folder + os.path.sep + 'f2005_full_' + target_label + os.path.sep + 'test.csv'
        if input_train_data is None:
            input_train_data = read_data(training_csv, target_label, target_label_name )
        else:
            input_train_data = input_train_data.append(read_data(training_csv, target_label, target_label_name))
        print('\ttotal test data :', input_train_data.shape, " adding ", target_label, " > ", target_label_name)

    print("\tpositive_instances per label :")
    positive_instances = input_train_data['target'].sum()
    print(positive_instances)

    print('\ttotal test data:', input_train_data.shape)

    return input_train_data

def balance_dataset(input_data, target_label, balance_n, seed):
    '''
        Balance a dataset towards the balance_n ratio with under-sampling the negative class (TNs) only.
    :param input_data:      The dataset to be balanced
    :param target_label:    The name of the column to use as the target variable (i.e. target in filtering)
    :param balance_n:       A number indicating the desired negative-to-positive-instances ratio.
    :param seed:            Random seed value
    :return:                The dataframe with balanced/under-sampled data
    '''
    if balance_n is not None :
        positive = input_data[input_data[target_label] == 1]
        negative = input_data[input_data[target_label] == 0]
        max_negative = len(positive) * balance_n
        if len(negative) > max_negative:
            negative = negative.sample(max_negative, random_state=seed)
            input_data = pd.concat([positive, negative], axis=0)
        else:
            print("\tNot enough negatives to undersample.")

    return input_data

def create_dataloader(dataset, batch_size, suffle = False):
    '''
    Create a torch DataLoader for the given dataset
    :param dataset: A TensorDataset
    :param batch_size: The batch size
    :param suffle: Whether to suffle the data or not
    :return: A DataLoader
    '''
    if suffle:
        # We'll take training samples in random order.
        sampler = RandomSampler(dataset)
    else:
        # For validation the order doesn't matter, so we'll just read them sequentially.
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,  # The training samples.
        sampler= sampler,  # Select batches accordingly
        batch_size=batch_size  # Trains with this batch size.
    )
    return dataloader

def fine_tune(settings, train_dataloader, device, validation_dataloader, pos_weight = None, class_freq = None, train_num = None, loss_func_name = None):
    '''
    Fine Tune a model.
    As long as our input data are sentences along with their corresponding labels, this task is similar to sentence classification task. For this reason, we will use the BertForSequenceClassification model.
    :param settings:
    :param target_labels:
    :param train_dataloader:
    :param device:
    :param validation_dataloader:
    :param pos_weight:
    :param class_freq:
    :param train_num:
    :param loss_func_name:
    :return:
    '''

    if not settings["pos_weight"]:
        pos_weight = None
    elif pos_weight is not None:
        pos_weight = pos_weight.to(device)

    modelName = settings["modelName"]
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    epochs = int(settings["epochs"])
    learning_rate = float(settings["learning_rate"])

    # if 'biobert_v1.1_pubmed' in modelName:
    #     # NOTE BioBERT model is being deprecated and does not work with the transformers current version.
    #     config = AutoConfig.from_pretrained(modelName, output_hidden_states=True, num_labels=len(target_labels))
    #     # print(str(config))
    #     model = BioBertForMultiLabelSequenceClassification2.from_pretrained(modelName, config=config)
    # else :
    # tested with: 'bert-base-uncased', 'allenai/scibert_scivocab_uncased', "allenai/biomed_roberta_base"
    target_labels = ['target']
    model = BertForMultiLabelSequenceClassification.from_pretrained(modelName, num_labels=len(target_labels),
                                                                    output_attentions=True)

    # Tell pytorch to run this model on the GPU.
    model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    # optimizer = AdamW(model.parameters(),
    #                   lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
    #                   eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
    #                 )

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, )

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Now the training loop
    # import random
    # import numpy as np

    # Store our loss and accuracy for plotting
    train_loss_set = []
    torch.cuda.empty_cache()

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # # Set the seed value all over the place to make this reproducible.
    # seed_val = int(settings["seed_val"])
    # random.seed(seed_val)
    # np.random.seed(seed_val)
    # torch.manual_seed(seed_val)
    # torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.

    # Measure the total training time for the whole run.
    start = time.time()
    best_model = None
    previous_model = None # The model trained one epoch prior to the currebt one.
    previous_to_best_model = None # The model trained one epoch prior to the best one.
    best_model_vaL_loss = None
    best_model_epoch = None
    early_stopping = False

    # For each epoch...
    for epoch_i in range(0, epochs):
        if not early_stopping:
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print('\t======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

            # Set our model to training mode (as opposed to evaluation mode)
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()
            wandb.watch(model)

            # Reset the total loss for this epoch.
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Train the data for one epoch
            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 500 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = round((time.time() - t0)/60, 2)

                    # Report progress.
                    print('\t  Batch {:>5,}  of  {:>5,}.    Elapsed: {:} m.'.format(step, len(train_dataloader), elapsed))

                # Add batch to GPU
                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Clear out the gradients (by default they accumulate)
                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                optimizer.zero_grad()
                # Forward pass
                # Perform a forward pass (evaluate the model on this training batch).
                # In PyTorch, calling `model` will in turn call the model's `forward`
                # function and pass down the arguments. The `forward` function is
                # documented here:
                # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
                # The results are returned in a results object, documented here:
                # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
                # Specifically, we'll get the loss (because we provided labels) and the
                # "logits"--the model outputs prior to activation.
                # Check that all tesors are in the same device:
                # print("b_input_ids.is_cuda", b_input_ids.is_cuda)
                # print("b_input_mask.is_cuda", b_input_mask.is_cuda)
                # print("b_labels.is_cuda", b_labels.is_cuda)
                # print("pos_weight.is_cuda", pos_weight.is_cuda)
                loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, pos_weight = pos_weight, class_freq = class_freq, train_num = train_num, loss_func_name = loss_func_name, device=device)

                train_loss_set.append(loss[0].item())
                # Backward pass
                # Perform a backward pass to calculate the gradients.
                loss[0].backward()
                # Update parameters and take a step using the computed gradient
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update tracking variables
                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                tr_loss += loss[0].item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                if step % 10000 == 0:
                    t = time.time()
                    print("\tTrain loss: {}".format(round(tr_loss / nb_tr_steps,2)))
                    print("\tTime: {}".format(round((t - start)/60, 2)))
                    wandb.log({"intermediate tr_loss per instance": round(tr_loss / nb_tr_examples, 2), 'epoch': (epoch_i+ 1)})
            print("\tEpoch train loss : {}".format(round(tr_loss / nb_tr_steps,2)))

            wandb.log({"total tr_loss per instance": round(tr_loss / len(train_dataloader),2), 'epoch': (epoch_i+ 1)})

            # Validation
            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids, b_input_mask, b_labels = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels = b_labels)
                    loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels = b_labels, pos_weight = pos_weight, class_freq = class_freq, train_num = train_num, loss_func_name = loss_func_name, device=device)

                # Get the loss and "logits" output by the model. The "logits" are the
                # output values prior to applying an activation function like the
                # softmax.
                # loss = result.loss
                # logits = result.logits

                # Move logits and labels to CPU
                # logits = logits[0].detach().cpu().numpy()
                # label_ids = b_labels.to('cpu').numpy()

                # Update tracking variables
                eval_loss += loss[0].item()
                nb_eval_examples += b_input_ids.size(0)
                nb_eval_steps += 1
                if step % 1000 == 0:
                    t = time.time()
                    print("\tValidation loss: {}".format(round(eval_loss / nb_eval_steps,2)))
                    # print("\tValidation acc: {}".format(round(eval_accuracy / len(validation_dataloader),2)))
                    print("\tTime: {}".format(round((t - start)/60, 2)))

            # print("\tEpoch validation acc: {}".format(eval_accuracy / len(validation_dataloader)))
            curent_val_loss = eval_loss / len(validation_dataloader)
            print("\tEpoch validation loss: {}".format(round(curent_val_loss,7)))
            wandb.log({"val_loss per instance": round(curent_val_loss,7), 'epoch': (epoch_i + 1)} )
            if best_model is None:
                best_model = copy.deepcopy(model)
                # If the best models is the one train on the first epoch, use this as previous as well.
                previous_to_best_model = copy.deepcopy(model)
                best_model_vaL_loss = curent_val_loss
                best_model_epoch = epoch_i + 1
            elif best_model_vaL_loss > curent_val_loss:
                print("\tModel of epoch ", best_model_epoch, " replaced by ", epoch_i + 1, " as best (", best_model_vaL_loss, " > ",curent_val_loss,")" )
                best_model = copy.deepcopy(model)
                previous_to_best_model = copy.deepcopy(previous_model)
                best_model_vaL_loss = curent_val_loss
                best_model_epoch = epoch_i + 1
            # else:
                # this epoch is worse than the previous, ignore further epochs
                # early_stopping = True

            previous_model = copy.deepcopy(model)
            #
            # [Optional] Save the fine-tuned model
            # if path_to_save is not None and settings["save_model"] and model_change:
            #     print('\tSave the fine-tunned model in ', path_to_save)
            #     # prev_path = path_to_save[:path_to_save.rfind(".")] + "_prev.pt"
            #     # torch.save(previous_to_best_model, prev_path)
            #     torch.save(model, path_to_save)

    end = time.time()
    t = end - start
    print("\tElapsed time: ", round(t/60, 7), "m")

    # # Report the final accuracy for this validation run.
    # avg_val_accuracy = eval_accuracy / len(validation_dataloader)
    # print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    wandb.log({"best_model_epoch": best_model_epoch})

    models ={}

    if settings["epoch_to_use"] == 'both':
        models['best'] = best_model
        models['prev'] = previous_to_best_model
    elif settings["epoch_to_use"] == 'more':
        models['best'] = best_model
        models['prev'] = previous_to_best_model
        models['current'] = model
    elif settings["epoch_to_use"] == 'previous':
        models['prev'] = previous_to_best_model
    elif settings["epoch_to_use"] == 'best':
        models['best'] = best_model

    return models

def print_token_stats(train_sentences, tokenizer):
    # find the maximum sentence length
    max_len = 0
    total_len = 0

    # For every sentence...
    for sent in train_sentences:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))
        total_len += len(input_ids)

    print('\tMax sentence length: ', max_len)
    print('\tAvg sentence length: ', total_len / len(train_sentences))

def print_tokenization_example(train_data,tokenizer):
    # a test sentece
    print('\tThis is an example of tokenization: ')
    t = train_data['text'].iloc[0]
    # Print the original sentence.
    print('\tOriginal: ', t)
    # Print the sentence split into tokens.
    print('\tTokenized: ', tokenizer.tokenize(t))
    # Print the sentence mapped to token ids.
    print('\tToken IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t)))

def predict(prediction_dataloader,model,device, prediction_threshold, verbose = True):
    # print('Predicting labels for {:,} test sentences...'.format(len(test_inputs)))
    print('\tPredicting labels for test sentences...')
    # print("prediction_dataloader", prediction_dataloader)
    # print("model", model)
    # Put model in evaluation mode
    # model.eval()

    # Tracking variables
    predictions = []
    true_labels = []
    flat_pred_probs = []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # result = model(b_input_ids,
            #               token_type_ids=None,
            #               attention_mask=b_input_mask,
            #               return_dict=True)
            logits = model(batch[0], token_type_ids=None, attention_mask=batch[1])[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            # print(logits)
            preds = []
            for l in logits:
                preds.append(l)
            # print(preds)
            sigmoid = torch.nn.Sigmoid()
            preds = sigmoid(torch.tensor(preds))
            # print(preds)
            preds = np.asarray(preds)
            # print(preds)
            # print("-")
            flat_pred_probs.append(preds)

            label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            preds2 = np.zeros(preds.shape)
            preds2[preds >= prediction_threshold] = 1
            predictions.append(preds2)
            true_labels.append(label_ids)

        if verbose and len(predictions)%10000 == 0:
            ratio = round((len(predictions) / len(prediction_dataloader)) * 100, 2)
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  len(predictions), " sentences processed (", ratio, "%)")
    # print(predictions)
    # print(true_labels)

    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)
    # For each sample, pick the label with the higher score.
    # flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # print(flat_predictions)

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)
    # print(flat_true_labels)

    print('\tDONE.')
    return flat_predictions, flat_true_labels, flat_pred_probs

def evaluation_report(flat_predictions, flat_true_labels, modelName, target_labels, dataset = ""):
    # Calculate the MCC
    mcc = f1_score(flat_true_labels, flat_predictions, average='macro')

    print('\n \t Total f1-score of ' + modelName + ' model for ' + str(
        target_labels) + ': %.3f \n' % mcc)
    # wandb.log({"macro f1-score " + dataset: round(mcc, 3)})

    print(classification_report(flat_true_labels, flat_predictions))
    return classification_report(flat_true_labels, flat_predictions, output_dict=True)

def imbalance_counter(train_y):
    # Calculate values about the imbalance in the training datasets useful for modifying the loss during training.
    # 1)    pos_weight tensor with weights to increase the impact of positive examples (used by BCEWithLogitsLoss)
    # 2)    class_freq array with the frequency of each label in the training datasets (used by ResampleLoss)
    # 3)    train_num the size of the training dataset (used by ResampleLoss)
    class_freq = torch.sum(train_y, 0) # i.e. positive count
    print("class_freq",class_freq)
    # total_count = len(train_y)
    train_num = train_y.shape[0] # i.e. total size of training dataset
    print("train_num",train_num)
    pos_weight = (train_num - class_freq)/ class_freq
    print("pos_weight", pos_weight)
    return torch.as_tensor(pos_weight, dtype=torch.float), class_freq, train_num
