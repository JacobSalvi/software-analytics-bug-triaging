import argparse
from pathlib import Path
from typing import AnyStr
import nltk
import marko
import marko.inline
import marko.md_renderer
import pandas as pd


def standardize_string(text: AnyStr) -> AnyStr:
    tokenized_text = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    words = [word for word in tokenized_text if word not in stopwords]
    stemmer = nltk.stem.PorterStemmer()
    stemmed_words = [stemmer.stem(word=word, to_lowercase=True) for word in words]
    return " ".join(stemmed_words)


def download_necessary_nltk_data():
    nltk.download('punkt_tab')
    nltk.download('stopwords')


def clean_text(block: marko.block.BlockElement):
    if isinstance(block, str):
        return
    for child in block.children:
        if isinstance(child, marko.inline.RawText):
            child.children = f" {standardize_string(child.children)} "
        else:
            clean_text(child)
    return


def parse_markdown(text: AnyStr) -> AnyStr:
    md_parser = marko.parser.Parser()
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
    renderer = marko.md_renderer.MarkdownRenderer()
    return renderer.render(ast)


def remove_pull_request(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df['pull_request'].notnull()]


def columns_parsing(df: pd.DataFrame) -> pd.DataFrame:
    df['body'] = df['body'].apply(parse_markdown)
    df['title'] = df['title'].apply(parse_markdown)
    return df


def pick_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ['url', 'title', 'body']]


def store_processed_data(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)


def data_slicer(df: pd.DataFrame, size: int) -> pd.DataFrame:
    return df.sample(n=size)


def main():
    download_necessary_nltk_data()
    parser = argparse.ArgumentParser("CleanUpIssues")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output", default='../output/cleaned_data.csv', type=Path, required=False)
    args = parser.parse_args()
    input_file: Path = args.file
    df = pd.read_json(input_file, lines=True)
    df = data_slicer(df, 100)
    df = remove_pull_request(df)
    df = pick_columns(df)
    columns_parsing(df)
    store_processed_data(df, args.output)


if __name__ == '__main__':
    main()
