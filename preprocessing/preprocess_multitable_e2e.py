import re
import sys
import glob
import pandas as pd
import numpy as np
import logging as logger
from io import StringIO
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
from datasets import disable_caching, DatasetDict
disable_caching()

def get_dataset(dataset_name):
    # pre-trainng stage 2 + 3
    if dataset_name == "multitable_pretraining":
        print("Processing MultiTable Pretraining dataset ...")
        valid_set = load_from_disk("/data/tir/projects/tir7/user_data/priyansk/MTabQA/data/spider/tokenized_spider_sql_valid.hf")
        synthetic_train_set = load_from_disk("/data/tir/projects/tir7/user_data/priyansk/MTabQA/data/pretraining_synthetic_dataset")
        spider_train_set = load_from_disk("/data/tir/projects/tir7/user_data/priyansk/MTabQA/data/spider/tokenized_spider_sql_train.hf")
        
        synthetic_train_set = synthetic_train_set.remove_columns(['db_name', 'query'])
        synthetic_train_set = synthetic_train_set.rename_column("question", "query")
        
        spider_train_set = spider_train_set.remove_columns(['question'])
        valid_set = valid_set.remove_columns(['question'])
        
        synthetic_train_set = synthetic_train_set.remove_columns(['input_ids', 'attention_mask', 'labels', 'decoder_input_ids'])
        spider_train_set = spider_train_set.remove_columns(['input_ids', 'attention_mask', 'labels', 'decoder_input_ids'])
        valid_set = valid_set.remove_columns(['input_ids', 'attention_mask', 'labels', 'decoder_input_ids'])

        train_set = concatenate_datasets([synthetic_train_set, spider_train_set])

        train_set = train_set.shuffle(seed=0)
        test_set = None
    
    elif dataset_name == "spider_nq":
        train_set = load_from_disk("/data/tir/projects/tir7/user_data/priyansk/MTabQA/data/spider/tokenized_spider_nq_train_with_answer.hf")["train"]
        valid_set = load_from_disk("/data/tir/projects/tir7/user_data/priyansk/MTabQA/data/spider/tokenized_spider_nq_valid_with_answer.hf")["train"]
        test_set = None

    elif dataset_name == 'atis':
        train_set = load_from_disk("/home/priyansk/qa/MultiTabQA/data/atis/atis_nq_train_with_answer.hf/")
        valid_set = load_from_disk("/home/priyansk/qa/MultiTabQA/data/atis/atis_nq_dev_with_answer.hf/")
        test_set = load_from_disk("/home/priyansk/qa/MultiTabQA/data/atis/atis_nq_test_with_answer.hf/")

        train_set = train_set.rename_column("question", "query")
        valid_set = valid_set.rename_column("question", "query")
        test_set = test_set.rename_column("question", "query")

    elif dataset_name == 'geoquery':
        train_set = load_from_disk("/home/priyansk/qa/MultiTabQA/data/geoquery/geoquery_nq_train_with_answer.hf/")
        valid_set = load_from_disk("/home/priyansk/qa/MultiTabQA/data/geoquery/geoquery_nq_dev_with_answer.hf/")
        test_set = load_from_disk("/home/priyansk/qa/MultiTabQA/data/geoquery/geoquery_nq_test_with_answer.hf/")

        train_set = train_set.rename_column("question", "query")
        valid_set = valid_set.rename_column("question", "query")
        test_set = test_set.rename_column("question", "query")

    else:
        raise Exception(f"Invalid dataset |{dataset_name}|")

    return train_set, valid_set, test_set


def clean_table(table):
    table = table.astype(str)
    table.fillna("", inplace=True)
    if len(table) > 0:
        # rename all duplicate column names
        if len(set(table.columns.to_list())) != len(table.columns):
            column_name_indices = defaultdict(list)
            for i, col_name in enumerate(table.columns):
                column_name_indices[col_name].append(i)
            new_column_names = [''] * len(table.columns)
            for col_name, indices in column_name_indices.items():
                if len(indices) > 1:
                    for j, index in enumerate(indices):
                        new_column_names[index] = f"{col_name} -{j + 1}"
                        # table.rename(inplace=True,columns={f'{index+1}col':f"{col_name}-{j+1}"})
                else:
                    new_column_names[indices[0]] = col_name
            table.columns = new_column_names

        # check all values of table and map them to str
        table = table.map(
            lambda x: pd.to_datetime(x, infer_datetime_format=True).strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(x, (pd.Timestamp, np.datetime64)) else x)
        table = table.map(lambda x: str(x))
        table.columns = table.columns.astype(str)
    return table

def prepare_table_query(
            tables,
            table_names,
            query,
    ):
    """
    This method can be used to linearize a table and add a corresponding query.
    """
    assert len(tables) == len(table_names), "Number of table names and tables must be same!"
    
    if query == "":
        logger.warning("You provide nothing to query with respect to the table.")

    query = query.replace("<>", "!=")
    separator = " " if query and tables else ""
    tables_with_separator = []
    
    for t, tn in zip(tables, table_names):
        tables_with_separator.append(f'<table_name> : {tn} {t}')
    
    tables_with_separator = " ".join(tables_with_separator)
    return (query + separator + tables_with_separator) if query else tables_with_separator


def parse_target(text):
    # table = text.split(" row ")
    table = re.split(r" row \d+ ", text)
    columns = table[0].split(": ")[1].split(" | ")
    columns = [x.strip() for x in columns]

    try:
        rows = [x.split(": ")[1].split(" | ") for x in table[1:]]
        rows = [[y.strip() for y in x] for x in rows]
    except:
        # print(text)
        # print(table[1:])
        # raise Exception("E")
        # logger.warning(f"Empty target table: {text}")
        rows = [""]*len(columns)

    df = pd.DataFrame(rows)
    df.columns = columns
    return df


def parse_source(text):
    
    sql_table = text.split(" <table_name> ")
    sql = sql_table[0].strip()

    tables = sql_table[1:]
    table_dfs = []
    table_names = []

    for table in tables:

        table_rows = table.split(" col : ")

        table_name = table_rows[0].split(": ")[1]
        table = table_rows[1]
        # table = table.split(" row ")
        table = re.split(r" row \d+ ", table)

        table_names.append(table_name)

        try:
            # first_colon = table[0].find(': ')
            # column_text = table[0][first_colon:]
            # columns = table[0].split(": ")[1].split(" | ")
            columns = table[0].split(" | ")
            columns = [x.strip() for x in columns]
        except:
            # print(text)
            # print(table[0])
            raise Exception("E")

        try:
            rows = [x.split(": ")[1].split(" | ") for x in table[1:]]
            rows = [[y.strip() for y in x] for x in rows]
        except:
            print("TEXT |", text)
            print("TABLE |", table[1:])
            raise Exception("E")

        df = pd.DataFrame(rows)
        try:
            df.columns = columns
        except:
            print("Text ", text)
            print("SQL ", sql)
            print("Table ", table)
            print("Col ", table[1])
            print(rows, columns)
            raise Exception("E")

        table_dfs.append(df)

    return sql, table_dfs, table_names



def format_multi_table(sample, table_type):
    # source = sample['source']
    # target = sample['target']

    # if source is None:
    #     print(sample.keys())
    #     print(sample)
    #     raise Exception("E")

    # query, tables, table_names = parse_source(source)
    # answer = parse_target(target)

    tables = sample["tables"]
    table_names = sample["table_names"]
    query = sample["query"] 
    answer = sample["answer"]

    if query is None or tables is None or table_names is None:
        query, tables, table_names = parse_source(sample['source'])

    if answer is None:
        answer = parse_target(sample['target'])
        

    tables = [clean_table(table) if isinstance(table, pd.core.frame.DataFrame) else clean_table(
            pd.read_json(StringIO(table), orient='split')) for table in tables]
        
    answer = clean_table(answer) if isinstance(answer, pd.core.frame.DataFrame) else clean_table(
        pd.read_json(StringIO(answer), orient='split'))

    if table_type == 'latex':
        tables_latex = [x.to_latex() for x in tables]
        answer = answer.to_latex()
        text = prepare_table_query(tables_latex, table_names, query)


    if table_type == 'html':
        tables_html = [x.to_html() for x in tables]
        answer = answer.to_html()
        text = prepare_table_query(tables_html, table_names, query)


    if table_type == 'markdown':
        tables_markdown = [x.to_markdown() for x in tables]
        answer = answer.to_markdown()
        text = prepare_table_query(tables_markdown, table_names, query)

    return {
        "source": text, "target": answer,
    }

def tokenize_sample(sample):
    input_encoding = tokenizer(sample[f'source'].strip().lower().replace('"', ''),
                               return_tensors="pt",
                               padding='max_length',
                               max_length=1024,
                               truncation='longest_first',
                               add_special_tokens=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            text=sample[f'target'].strip().lower().replace('"', ''),
            add_special_tokens=True,
            return_tensors="pt",
            padding='max_length',
            max_length=1024,
            truncation='longest_first',
        )
    
    return {
        "input_ids": input_encoding["input_ids"],
        "attention_mask": input_encoding["attention_mask"],
        "labels": labels['input_ids'],
    }



def tokenize_batch(samples):
    input_encoding = tokenizer([x.strip().lower().replace('"', '') for x in samples[f'source']],
                               return_tensors="pt",
                               padding='max_length',
                               max_length=1024,
                               truncation='longest_first',
                               add_special_tokens=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            text=[x.strip().lower().replace('"', '') for x in samples[f'target']],
            add_special_tokens=True,
            return_tensors="pt",
            padding='max_length',
            max_length=1024,
            truncation='longest_first',
        )
    
    return {
        "input_ids": input_encoding["input_ids"],
        "attention_mask": input_encoding["attention_mask"],
        "labels": labels['input_ids'],
    }


dataset_name = sys.argv[1]
shard_idx = int(sys.argv[2])
end_idx = int(sys.argv[3])
model_name = sys.argv[4]
table_type = sys.argv[5]

tokenizer = AutoTokenizer.from_pretrained(model_name)


save_file = f"kpriyanshu256/MultiTabQA-{dataset_name}-{model_name.replace('/', '-')}_train-html-{shard_idx}"

print(f'Dataset {dataset_name}')
print("Processing shard ", shard_idx, end_idx)
print("Will save to ", save_file)

train_dataset, valid_dataset, test_dataset = get_dataset(dataset_name)

end_idx = min(end_idx, len(train_dataset))

train_dataset = train_dataset.select([*range(shard_idx, end_idx)])

print(train_dataset)

NUM_PROCS = 32

train_dataset = train_dataset.map(format_multi_table,
                                    num_proc=NUM_PROCS, 
                                    desc="Processing", 
                                    remove_columns=['answer', 'query', 'tables', 'table_names', 'source', 'target'],
                                    fn_kwargs={"table_type": table_type},
                                    )

# train_dataset = train_dataset.map(tokenize_batch, 
#                                 batched=True, 
#                                 batch_size=2,
#                                 remove_columns=['source', 'target'], desc='Tokenizing')

train_dataset = train_dataset.map(tokenize_sample, 
                                num_proc=NUM_PROCS//4, 
                                remove_columns=['source', 'target'], 
                                desc='Tokenizing')


train_dataset.push_to_hub(save_file)
print(f"Saved dataset to {save_file}")