import re
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
import argparse

from code_names_mapping import mapping
from papers_old_list import old_list


import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


INDENT = "    "


def get_all_urls(src):
    with open(src, mode='r', encoding='utf8') as f:
        text = f.read()
        url_list = re.findall(pattern=r"https://arxiv\.org/abs/\d+\.\d+", string=text)
    return url_list


def scrape_arxiv(url):
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    # generate links
    idx = url.split('/')[-1]
    url_abs = url
    url_pdf = f"https://arxiv.org/pdf/{idx}.pdf"
    url_van = f"https://www.arxiv-vanity.com/papers/{idx}/"
    # get title
    title = soup.find("h1", class_="title mathjax").text.strip()
    title = re.sub(pattern=r"Title: *", repl="", string=title)
    # get year
    year = soup.find("div", class_="dateline").text.strip()
    year = re.findall(pattern=r"Submitted on \d+ \w{3} \d{4}", string=year)[0]
    year = re.sub(pattern=r"Submitted on ", repl="", string=year)
    year = year.split(' ')
    year[0] = '0' * (2-len(year[0])) + year[0]
    year[2] = '`' + year[2] + '`'
    year = ' '.join(year)
    # get authors
    authors = soup.find("div", class_="authors").text.strip()
    authors = re.sub(pattern=r"Authors: *", repl="", string=authors)
    # get abstract
    abstract = soup.find("blockquote", class_="abstract mathjax").text.strip()
    abstract = re.sub(pattern=r"Abstract: *", repl="", string=abstract)
    abstract = re.sub(pattern=r"\n+", repl=" ", string=abstract)
    # define string
    string = ""
    string += f"* [[{mapping.get(title, title)}]({url_abs})]" + '\n'
    string += INDENT + f"[[pdf]({url_pdf})]" + '\n'
    string += INDENT + f"[[vanity]({url_van})]" + '\n'
    string += INDENT + "* Title: " + title + '\n'
    string += INDENT + "* Year: " + year + '\n'
    string += INDENT + "* Authors: " + authors + '\n'
    string += INDENT + "* Abstract: " + abstract + '\n'
    return string


def scrape_mdpi(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        # get title
        title = soup.find("h1", class_="title hypothesis_container").text.strip()
        # get year
        year = soup.find("div", class_="pubhistory").text.strip()
        year = year.split('/')[0].split(':')[1].strip().split(' ')
        year[0] = '0' * (2-len(year[0])) + year[0]
        year[1] = year[1][:3]
        year[2] = '`' + year[2] + '`'
        year = ' '.join(year)
        # get authors
        authors = soup.find_all("a", class_="sciprofiles-link__link")
        authors = ', '.join([author.text for author in authors])
        # get abstract
        abstract = soup.find("div", class_="html-p").text.strip()
        # define string
        string = ""
        string += f"* [[{mapping.get(title, title)}]({url})]" + '\n'
        string += INDENT + "* Title: " + title + '\n'
        string += INDENT + "* Year: " + year + '\n'
        string += INDENT + "* Authors: " + authors + '\n'
        string += INDENT + "* Abstract: " + abstract + '\n'
        return string
    else:
        logging.error(f"Failed to get response from {url}.")
        return ""


def scrape_single(url):
    if url.startswith("https://arxiv.org"):
        return scrape_arxiv(url)
    if url.startswith("https://www.mdpi.com/"):
        return scrape_mdpi(url)
    else:
        logging.error(f"No scraper implemented for {url}.")
        return ""


def main(url_dict, src=None, dst=None):
    """
    Arguments:
        url_dict (dict): dictionary of lists of urls to be scrapped.
        src (str): source file to extract urls from.
        dst (str): will be replaced with temp files if `url_dict` is not None.
    """
    if url_dict is None:
        assert src is not None
        logging.info(f"Argument `url_dict` not provided. Extracting urls from {src}.")
        url_dict = get_all_urls(src)
    else:
        assert type(url_dict) == dict, f"{type(url_dict)=}"
        logging.info(f"Argument `url_dict` provided. Suppressing arguments `src` and `dst`.")
        for group in url_dict:
            url_list = url_dict[group]
            logging.info(f"Processing group '{group}'." + (
                " Nothing given." if len(url_list) == 0 else f" {len(url_list)} urls found."))
            url_set = set(url_list)
            if len(url_set) != len(url_list):
                logging.info(f"Provided list contains duplicates. Reduced to {len(url_set)} papers.")
            dst = f"scripts/scraper/temp_{group}.md"
            with open(dst, mode='w', encoding='utf8') as f:
                for idx, url in enumerate(url_set):
                    logging.info(f"[{idx+1}/{len(url_set)}] Scraping {url}")
                    f.write(scrape_single(url))
    logging.info(f"Process terminated.")


if __name__ == "__main__":
    main(url_dict=old_list)
