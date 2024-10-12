import argparse
import re
from pathlib import Path
from typing import AnyStr
import nltk
from nltk.data import find
import marko
import marko.inline
import marko.md_renderer
import pandas as pd
import demoji
import swifter

from src.DataProvider import DataProvider


def standardize_string(text: AnyStr) -> AnyStr:
    text = demoji.replace(text, repl="")
    html_tags_regex = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text= re.sub(html_tags_regex, '', text)
    tokenized_text = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    words = [word for word in tokenized_text if word not in stopwords]
    stemmer = nltk.stem.PorterStemmer()
    stemmed_words = [stemmer.stem(word=word, to_lowercase=True) for word in words]
    return " ".join(stemmed_words)


def download_necessary_nltk_data():
    try:
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        print("'punkt' downloaded")

    try:
        find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        print("'stopwords' downloaded")


def clean_text(block: marko.block.BlockElement):
    if isinstance(block, str):
        return
    for child in block.children:
        if isinstance(child, marko.inline.InlineHTML):
            block.children.remove(child)
            continue
        if isinstance(child, marko.inline.RawText):
            child.children = f"{standardize_string(child.children)}"
        else:
            clean_text(child)
    return


def parse_markdown(text: AnyStr, md_parser=None, renderer=None) -> AnyStr:
    if text is None:
        return ""
    ast = md_parser.parse(text)
    for child in ast.children:
        if isinstance(child, marko.block.Heading):
            clean_text(child)
        if isinstance(child, marko.block.Paragraph):
            clean_text(child)
        if isinstance(child, marko.block.List):
            clean_text(child)
        if isinstance(child, marko.block.CodeBlock):
            pass

    return renderer.render(ast)


def remove_pull_request(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df['pull_request'].notnull()]

def columns_parsing(df: pd.DataFrame) -> pd.DataFrame:
    md_parser = marko.parser.Parser()
    renderer = marko.md_renderer.MarkdownRenderer()
    # swifter to parallelize the process
    df['body'] = df['body'].swifter.apply(lambda text: parse_markdown(text, md_parser, renderer))
    df['title'] = df['title'].swifter.apply(lambda text: parse_markdown(text, md_parser, renderer))
    return df

def pick_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ['url', 'title', 'body']]


def store_processed_data(df: pd.DataFrame, output_path: Path):
    df.to_csv(output_path, index=False)


def data_slicer(df: pd.DataFrame, size: int) -> pd.DataFrame:
    return df.sample(n=size)


def process_data(input_file: Path, output_path: Path):
    download_necessary_nltk_data()

    # Check if input_file is provided, otherwise use the raw JSON data
    if input_file is not None:
        df = pd.read_json(input_file, lines=True)  # This is for file-based input
    else:
        # If get_raw() returns a list of JSON objects, use pd.DataFrame()
        raw_data = DataProvider.get_raw()
        df = pd.DataFrame(raw_data)

    # Proceed with the remaining processing steps
    df = remove_pull_request(df)
    df = pick_columns(df)
    columns_parsing(df)
    store_processed_data(df, output_path)


def main():
    download_necessary_nltk_data()
    parser = argparse.ArgumentParser("CleanUpIssues")
    parser.add_argument("--file", type=Path, default=None, help="Path to the raw json data, if not specified, it will use the default data")
    parser.add_argument("--output", default='../data/cleaned_parsed_issues.csv', type=Path, required=False,  help="Path to the output json data")
    args = parser.parse_args()
    process_data(args.file, args.output)


if __name__ == '__main__':
    main()