import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../DB_datas/output.csv')[['본문','요약']]
df.dropna(inplace = True)
contexts, summaries = df['본문'].tolist(), df['요약'].tolist()
train_contexts, train_summaries , test_contexts, test_summaries = train_test_split(contexts, summaries, random_state = 25)

import torch
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from transformers import AutoTokenizer
from dataset import MyDataset



def train_model():

    tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")
    model = AutoModelForSeq2SeqLM.from_pretrained("digit82/kobart-summarization")

    

    # Cut sequences longer than the maximum length of the model and padded to the maximum length
    train_inputs = tokenizer(train_contexts, truncation=True, padding=True, max_length=128)
    train_labels = tokenizer(train_summaries, truncation=True, padding=True, max_length=64)    
    # dataset 
    train_dataset = MyDataset(train_inputs, train_labels)

    # Cut sequences longer than the maximum length of the model and padded to the maximum length
    test_inputs = tokenizer(test_contexts, truncation=True, padding=True, max_length=128)
    test_labels = tokenizer(test_summaries, truncation=True, padding=True, max_length=64)    
    # dataset 
    train_dataset = MyDataset(test_inputs, test_labels)
    test_dataset = MyDataset(test_inputs, test_labels)


    # traing arguments
    training_args = TrainingArguments(
        output_dir='./new_results',          
        num_train_epochs=50,                 
        per_device_train_batch_size=16,      # batch size
        per_device_eval_batch_size=64,       # batch size
        warmup_steps=500,                    # Used to incrementally increase the learning rate
        weight_decay=0.01,                   # strength of weight decay
        logging_dir='./new_logs',            
        logging_steps=5,                    # Learning Loss Tracking

        # Early stopping
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',  # Specify the metric you want to use for early stopping
        greater_is_better=False,  # Set to True if the metric should increase for better performance

       # Early stopping parameters
        evaluation_strategy="steps",  # Set to "steps" to perform evaluation at fixed intervals
        eval_steps=5,  # Number of training steps between evaluations
        save_total_limit=3,  # Limit the total number of checkpoints to save
    )


    # Create the Trainer and train
    trainer = Trainer(
        model=model,                         
        args=training_args,                 
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # Specify the validation dataset
    )


    
    trainer.train()

    # Save the newly trained model weights
    torch.save(model.state_dict(), 'weights_new.ckpt')