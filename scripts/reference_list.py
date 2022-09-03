import os
from serpapi import GoogleSearch
import re


def process_line(line):
    line = line.strip()
    line = re.sub(r'[^\x00-\x7F]+', ' ', line)
    line = re.sub(f' +', ' ', line)
    return line


def search(paper):
    params = {
        "engine": "google_scholar_cite",
        "api_key": "858506dd1aab6dc4f01da2b1cda760830bdc89456eacc02e7ec1f5543af326cb",
        "q": paper,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    citation = results["citations"][0]['snippet']
    return citation


def get_reference_list(input_filename):
    with open(os.path.join('_references', input_filename), 'r') as f:
        reference_list = [line.strip() for line in f.readlines() if len(line.strip()) != 0 and line.strip() not in ['References', 'REFERENCES']]
    return reference_list


def rewrite(reference_list, output_filename):
    counter = 1
    failure_list = []
    with open(os.path.join('test', output_filename), 'w') as f:
        f.write('References\n')
        for paper in reference_list:
            flag = ''
            try:
                citation = search(paper)
            except:
                print(f"[ERROR] {paper}")
                print(f"Saving original text.")
                citation = paper
                failure_list.append(paper)
                flag = '[WARNING]'
            citation = re.sub('\u200f', '', citation)
            print(citation)
            print()
            f.write(flag + f'[{counter}] ' + citation + '\n')
            counter += 1
    print(f"All done. {len(failure_list)} failed.")
    print(failure_list)


if __name__ == "__main__":
    input_filename = 'EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.md'
    output_filename = input_filename
    rewrite(get_reference_list(input_filename), output_filename)
