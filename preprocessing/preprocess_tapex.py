import sys
import glob
import pandas as pd
import numpy as np
import logging as logger
from io import StringIO
from datasets import load_from_disk, concatenate_datasets

def get_dataset(dataset_name):
    # pre-training stage 1
    if dataset_name == "tapex_pretraining":
        print("Processing Tapex Pretrainng dataset ...")
        train_set =  concatenate_datasets([load_from_disk(path) for path in glob.glob("/data/tir/projects/tir7/user_data/priyansk/MTabQA/data/tapex_pretraining/preprocessed_train_*.hf")])
        valid_set = load_from_disk("/data/tir/projects/tir7/user_data/priyansk/MTabQA/data/tapex_pretraining/preprocessed_valid.hf")
        test_set = None
        print(train_set)
    # pre-trainng stage 2 + 3
    elif dataset_name == "multitable_pretraining":
        print("Processing MultiTable Pretraining dataset ...")
        valid_set = load_from_disk("/data/tir/projects/tir7/user_data/priyansk/MTabQA/data/spider/sql/spider_sql_valid.hf")
        synthetic_train_set = load_from_disk("/data/tir/projects/tir7/user_data/priyansk/MTabQA/data/pretraining_synthetic_dataset")
        spider_train_set = load_from_disk("/data/tir/projects/tir7/user_data/priyansk/MTabQA/data/spider/sql/spider_sql_train.hf")
        
        synthetic_train_set = synthetic_train_set.remove_columns(['db_name'])
        
        train_set = concatenate_datasets([synthetic_train_set, spider_train_set])
        train_set = train_set.shuffle(seed=args.seed)
        test_set = None
        print(f"Training with {len(train_set)} samples")
    else:
        raise Exception("Invalid dataset")

    try:
        train_set = train_set.remove_columns(['input_ids', 'attention_mask', 'labels', 'decoder_input_ids'])
        valid_set = valid_set.remove_columns(['input_ids', 'attention_mask', 'labels', 'decoder_input_ids'])
        if test_set is not None:
            test_set = test_set.remove_columns(['input_ids', 'attention_mask', 'labels', 'decoder_input_ids'])

    except:
        print("Dataset not tokenized")

    return train_set, valid_set, test_set

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
    # tables_with_separator = " ".join(tables)
    tables_with_separator = []
    
    for t, tn in zip(tables, table_names):
        tables_with_separator.append(f'<table_name> : {tn} {t}')
    tables_with_separator = " ".join(tables_with_separator)

    return (query + separator + tables_with_separator) if query else tables_with_separator


def parse_target(text):
    table = text.split(" row ")
    columns = table[0].split(": ")[1].split(" | ")
    columns = [x.strip() for x in columns]

    try:
        rows = [x.split(" : ")[1].split(" | ") for x in table[1:]]
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
    sql_table = text.split(" col : ")

    sql = sql_table[0].strip()

    table = sql_table[1].split(" row ")
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
        rows = [x.split(" : ")[1].split(" | ") for x in table[1:]]
        rows = [[y.strip() for y in x] for x in rows]
    except:
        # print(text)
        # print(table[1:])
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

    return sql, df



def format_single_table(sample):
    source = sample['source']
    target = sample['target']

    query, table = parse_source(source)
    answer = parse_target(target)

    tables = [table]

    # tables = [clean_table(table) if isinstance(table, pd.core.frame.DataFrame) else clean_table(
    #         pd.read_json(StringIO(table), orient='split')) for table in tables]
    
    
    # answer = clean_table(answer) if isinstance(answer, pd.core.frame.DataFrame) else clean_table(
    #     pd.read_json(StringIO(answer), orient='split'))


    tables_latex = [x.to_latex() for x in tables]
    answer_latex = answer.to_latex()

    tables_html = [x.to_html() for x in tables]
    answer_html = answer.to_html()

    tables_markdown = [x.to_markdown() for x in tables]
    answer_markdown = answer.to_markdown()

    text_latex = prepare_table_query(tables_latex, ['table'], query)
    text_html = prepare_table_query(tables_html, ['table'], query)
    text_markdown = prepare_table_query(tables_markdown, ['table'], query)

    return {
        "source_latex": text_latex, "target_latex": answer_latex,
        "source_html": text_html, "target_html": answer_html,
        "source_markdown": text_markdown, "target_markdown": answer_markdown,
    }


# dataset_name = "spider_nq"
dataset_name = sys.argv[1]

train_dataset, valid_dataset, test_dataset = get_dataset(dataset_name)


# DB = 10000
# train_dataset = train_dataset.select([*range(DB)])
# valid_dataset = valid_dataset.select([*range(DB)])
# if test_dataset is not None:
#     test_dataset = test_dataset.select([*range(DB)])

NUM_PROCS = 16

train_dataset = train_dataset.map(format_single_table, num_proc=NUM_PROCS, desc="Processing Train")
save_file = f"/data/tir/projects/tir7/user_data/priyansk/QA-DS/{dataset_name}/table-train"
print("Saving ", save_file)
train_dataset.save_to_disk(save_file)

valid_dataset = valid_dataset.map(format_single_table, num_proc=NUM_PROCS, desc="Processing Valid")
save_file = f"/data/tir/projects/tir7/user_data/priyansk/QA-DS/{dataset_name}/table-valid"
print("Saving ", save_file)
valid_dataset.save_to_disk(save_file)


if test_dataset is not None:
    test_dataset = test_dataset.map(format_single_table, num_proc=NUM_PROCS, desc="Processing Test")
    save_file = f"/data/tir/projects/tir7/user_data/priyansk/QA-DS/{dataset_name}/table-test"
    print("Saving ", save_file)

    test_dataset.save_to_disk(save_file)
    