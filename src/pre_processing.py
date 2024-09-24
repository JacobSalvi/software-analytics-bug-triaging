from typing import AnyStr
import nltk
import marko
import marko.inline
import marko.md_renderer

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
    for child in block.children:
        if isinstance(child, marko.inline.RawText):
            child.children = f" {standardize_string(child.children)} "
        else:
            clean_text(child)
    return

def parse_markdown(text: AnyStr) -> AnyStr:
    md_parser = marko.parser.Parser()
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



def main():
    sentence = ["program", "programs", "programmer", "Programming", "programmers"]
    sentence = " ".join(sentence)
    download_necessary_nltk_data()
    print(standardize_string(sentence))
    f = open("../example2.md")
    content = f.read()
    cleaned_markdown = parse_markdown(content)
    f = open("../robe.md", 'w')
    f.write(cleaned_markdown)

if __name__ == '__main__':
    main()
