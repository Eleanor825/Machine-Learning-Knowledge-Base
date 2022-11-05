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


def main(url_lists, src_filepath, dst_filepath):
    """
    Arguments:
        url_lists (dict): five lists of urls to be scrapped.
        src_filepath (str):
        dst_filepath (str): will be replaced with temp files if url_lists is not None.
    """
    if url_lists is None:
        print(f"[INFO] Argument `url_lists` not provided. Extracting urls from {src_filepath}.")
        url_lists = get_all_urls(src_filepath)
    else:
        assert type(url_lists) == dict, type(url_list)
        print(f"[INFO] Argument `dst_filepath` suppressed. Using temp.md.")
    for group in url_lists:
        print(f"[INFO] Processing {group}." + (
            " Nothing given." if len(url_lists[group]) == 0 else f" {len(url_lists[group])} urls found."))
        dst_filepath = f"scripts/scrape-paper-info/temp_{group}.md"
        with open(dst_filepath, mode='w', encoding='utf8') as f:
            for url in url_lists[group]:
                print(f"[INFO] Scraping {url}")
                f.write(scrape_single(url))
    print(f"[INFO] Process terminated.")


if __name__ == "__main__":
    main(url_lists={
        'C': [
        ],
        'D': [
        ],
        'S': [
        ],
        'T': [
        ],
        'G': [
            "https://arxiv.org/abs/1406.2661",
        ],
    },
        src_filepath="papers-cv/tasks/detection_2D.md",
        dst_filepath="scripts/scrape-paper-info/raw_detection_2D.md",
    )
