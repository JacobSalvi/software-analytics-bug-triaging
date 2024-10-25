import argparse
import re
import string
from pathlib import Path
from typing import AnyStr, List
import pandas as pd
import demoji
import swifter
import nltk
from marko.md_renderer import MarkdownRenderer
from nltk.data import find
from nltk.stem import WordNetLemmatizer
import unicodedata
import contractions
import marko
from marko.inline import InlineHTML
from marko.block import  List

from src.processing.assignees_processing import filter_assignee_data
from src.utils import utils

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

    try:
        find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        print("'wordnet' downloaded")


LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

HTML_TAGS_REGEX = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
URL_REGEX = re.compile(r'http[s]?://\S+|www\.\S+')
EMAIL_REGEX = re.compile(r'\S+@\S+')
MENTIONS_REGEX = re.compile(r'@\w+')
HASHTAGS_REGEX = re.compile(r'#\w+')
REPEATED_CHARS_REGEX = re.compile(r'(.)\1{2,}')
NUMBERS_REGEX = re.compile(r'\d+')
NON_WORD_REGEX = re.compile(r'[^\w\s]')
MODULES_REGEX = re.compile(r'(\w+\.)+\w')


class MdRenderer(marko.md_renderer.MarkdownRenderer):
    def render_emphasis(self, element: marko.inline.Emphasis) -> str:
        return f" *{self.render_children(element)}* "

    def render_strong_emphasis(self, element: marko.inline.StrongEmphasis) -> str:
        return f" **{self.render_children(element)}** "


def standardize_string(text: AnyStr) -> AnyStr:
    text = demoji.replace(text, repl="")
    text = re.sub(HTML_TAGS_REGEX, '', text)
    text = re.sub(URL_REGEX, '', text)
    text = re.sub(EMAIL_REGEX, '', text)
    text = unicodedata.normalize('NFC', text)
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(MENTIONS_REGEX, '', text)
    text = re.sub(HASHTAGS_REGEX, '', text)
    text = re.sub(REPEATED_CHARS_REGEX, r'\1\1', text)
    text = re.sub(NUMBERS_REGEX, '', text)
    text = ' '.join(text.split())
    text = text.encode('ascii', 'ignore').decode()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    tokens: List[AnyStr] = text.split(" ")
    translation = str.maketrans('', '', string.punctuation)
    tokens = list(
        map(lambda el: el if el.lower() in ["c#", "f#"] or re.match(MODULES_REGEX, el) else el.translate(translation),
            filter(lambda token: len(token) <= 40, tokens)))
    text = " ".join(tokens)
    tokenized_text = nltk.word_tokenize(text)
    words = [word for word in tokenized_text if word not in STOPWORDS]
    lemmatized_words = [LEMMATIZER.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


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
    df['body'] = df['body'].fillna('').apply(lambda x: x.strip())
    df['title'] = df['title'].fillna('').apply(lambda x: x.strip())
    return df

def pick_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ['id', 'number', 'url', 'title', 'body', 'assignee', "labels"]]

def store_processed_data(df: pd.DataFrame, output_path: Path):
    df.to_csv(output_path, index=False)


def data_slicer(df: pd.DataFrame, size: int) -> pd.DataFrame:
    return df.sample(n=size)


def process_input(input_file: Path, output_path: Path):
    df = pd.read_json(input_file, lines=True)
    df = process_data(df)
    store_processed_data(df, output_path)

def process_data(df: pd.DataFrame) -> pd.DataFrame:
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
    parser.add_argument("--output", default=utils.data_dir().joinpath("cleaned_parsed_issues.csv"),
                        type=Path, required=False, help="Path to the output json data")
    args = parser.parse_args()
    process_input(args.file, args.output)


if __name__ == '__main__':
    main()