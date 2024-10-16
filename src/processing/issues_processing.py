import argparse
import re
import string
from pathlib import Path
from typing import AnyStr
import nltk
from nltk.data import find
import marko
import marko.inline
import marko.md_renderer
import pandas as pd
import demoji
import swifter #do not remove!
from pandas.core.interchange.dataframe_protocol import DataFrame
from src.processing.assignees_processing import filter_assignee_data


class MdRenderer(marko.md_renderer.MarkdownRenderer):
    def render_emphasis(self, element: marko.inline.Emphasis) -> str:
        return f" *{self.render_children(element)}* "

    def render_strong_emphasis(self, element: marko.inline.StrongEmphasis) -> str:
        return f" **{self.render_children(element)}** "


def standardize_string(text: AnyStr) -> AnyStr:
    text = demoji.replace(text, repl="")
    html_tags_regex = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(html_tags_regex, '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokenized_text = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    words = [re.sub(r'[^\w\s]', '', word) for word in tokenized_text if word not in stopwords]
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
        if isinstance(child, marko.inline.Image):
            block.children.remove(child)
            continue
        if isinstance(child, marko.inline.Link):
            block.children.remove(child)
            continue
        if isinstance(child, marko.inline.AutoLink):
            block.children.remove(child)
            continue
        if isinstance(child, marko.inline.RawText):
            child.children = f"{standardize_string(child.children)}"
        else:
            clean_text(child)
    return


def parse_markdown(text: AnyStr, md_parser=None, renderer=None) -> AnyStr:
    try:
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
    
    except Exception as e:
        print(f"Exception: {e}")  
        print(f"Problem: \n {text}")  
        return ""  


def remove_pull_request(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df['pull_request'].notnull()]

def columns_parsing(df: pd.DataFrame) -> pd.DataFrame:
    md_parser = marko.parser.Parser()
    renderer = MdRenderer()
    # swifter to parallelize the process
    df['body'] = df['body'].swifter.apply(lambda text: parse_markdown(text, md_parser, renderer))
    df['title'] = df['title'].swifter.apply(lambda text: parse_markdown(text, md_parser, renderer))
    return df

def pick_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ['id','number', 'url', 'title', 'body','assignee', "labels"]]

def store_processed_data(df: pd.DataFrame, output_path: Path):
    df.to_csv(output_path, index=False)


def data_slicer(df: pd.DataFrame, size: int) -> pd.DataFrame:
    return df.sample(n=size)


def process_input(input_file: Path, output_path: Path):
    df = pd.read_json(input_file, lines=True)
    df = process_data(df)
    store_processed_data(df, output_path)

def process_data(df: DataFrame):
    download_necessary_nltk_data()
    df = remove_pull_request(df)
    df = pick_columns(df)
    df["labels"] = df["labels"].apply(lambda arr: [el.get("name") for el in arr])
    df = columns_parsing(df)
    df = filter_assignee_data(df)
    return df

def main():
    download_necessary_nltk_data()
    parser = argparse.ArgumentParser("Clean up issues")
    parser.add_argument("--file", type=Path, required=True, help="Path to the raw json data")
    parser.add_argument("--output", default='../data/cleaned_parsed_issues.csv', type=Path, required=False,  help="Path to the output json data")
    args = parser.parse_args()
    process_input(args.file, args.output)


if __name__ == '__main__':
    main()
