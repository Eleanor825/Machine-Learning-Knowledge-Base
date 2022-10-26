import re
from bs4 import BeautifulSoup
from urllib.request import urlopen


def get_all_urls(src_filepath):
    with open(src_filepath, mode='r', encoding='utf8') as f:
        text = f.read()
        url_list = re.findall(pattern=r"https://arxiv\.org/abs/\d+\.\d+", string=text)
    return url_list


def scrape_single(url: str):
    INDENT = "    "
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    # generate links
    idx = url.split('/')[-1]
    url_abs = url
    url_pdf = f"https://arxiv.org/pdf/{idx}.pdf"
    url_van = f"https://www.arxiv-vanity.com/papers/{idx}/"
    # extract title
    title = soup.find("h1", class_="title mathjax").text.strip()
    title = re.sub(pattern=r"Title: *", repl="", string=title)
    # extract year
    year = soup.find("div", class_="dateline").text.strip()
    year = re.findall(pattern=r"Submitted on \d+ \w{3} \d{4}", string=year)[0]
    year = re.sub(pattern=r"Submitted on ", repl="", string=year)
    year = year.split(' ')
    year[0] = '0' * (2-len(year[0])) + year[0]
    year[2] = '`' + year[2] + '`'
    year = ' '.join(year)
    # extract authors
    authors = soup.find("div", class_="authors").text.strip()
    authors = re.sub(pattern=r"Authors: *", repl="", string=authors)
    # extract abstract
    abstract = soup.find("blockquote", class_="abstract mathjax").text.strip()
    abstract = re.sub(pattern=r"Abstract: *", repl="", string=abstract)
    abstract = re.sub(pattern=r"\n+", repl=" ", string=abstract)
    # define string
    string = ""
    string += f"* [[{title}]({url_abs})]" + '\n'
    string += INDENT + f"[[pdf]({url_pdf})]" + '\n'
    string += INDENT + f"[[vanity]({url_van})]" + '\n'
    string += INDENT + "* Title: " + title + '\n'
    string += INDENT + "* Year: " + year + '\n'
    string += INDENT + "* Authors: " + authors + '\n'
    string += INDENT + "* Abstract: " + abstract + '\n'
    return string


def main(url_list, src_filepath, dst_filepath):
    if url_list is None:
        print(f"[INFO] url_list not provided, extracting urls from {src_filepath}.")
        url_list = get_all_urls(src_filepath)
    with open(dst_filepath, mode='w', encoding='utf8') as f:
        print(f"[INFO] {len(url_list)} papers in total.")
        for url in url_list:
            print(f"[INFO] Scraping {url}")
            f.write(scrape_single(url))
        print(f"[INFO] All results saved to {dst_filepath}.")


if __name__ == "__main__":
    main(
        url_list=[
            "https://arxiv.org/abs/1502.07411",
            "https://arxiv.org/abs/1411.6387",
            "https://arxiv.org/abs/1411.4734",
            "https://arxiv.org/abs/1406.2283",
        ],
        src_filepath="papers-cv/architectures/CNN.md",
        dst_filepath="scripts/scrape-paper-info/raw_CNN.md",
    )
