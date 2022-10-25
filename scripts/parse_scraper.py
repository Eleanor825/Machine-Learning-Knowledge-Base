import csv


def abs_to_pdf(url_abs):
    return f"https://arxiv.org/pdf/{url_abs.split('/')[-1]}.pdf"


def parse_title(title):
    parsed = ':'.join(title.split(':')[1:])
    assert "Title:" + parsed == title
    return parsed


def parse_year(year):
    parsed = year
    parsed = parsed.replace('[', '')
    parsed = parsed.replace(']', '')
    parsed = parsed.split(' ')[2:5]
    if len(parsed[0]) == 1:
        parsed[0] = '0' + parsed[0]
    parsed[-1] = '`' + parsed[-1] + '`'
    parsed = ' '.join(parsed)
    assert len(parsed) == 13
    return parsed


def parse_authors(authors):
    parsed = authors.split(':')[1]
    assert "Authors:" + parsed == authors
    return parsed


def parse_abstract(abstract):
    parsed = abstract
    parsed = parsed.replace('\n', ' ')
    parsed = ':'.join(parsed.split(':')[1:])
    parsed = parsed.replace('  ', ' ')
    parsed = parsed.strip(' ')
    return parsed


if __name__ == "__main__":

    filename = "scripts/extract-paper-info-6.csv"
    string = ""

    with open(filename, mode='r') as f:
        reader = csv.reader(f)
        reader = list(reader)
        reader = reader[1:]  # skip title line
        for row in reader:
            url_abs = row[1]
            url_pdf = abs_to_pdf(url_abs)
            title = parse_title(row[2])
            year = parse_year(row[5])
            authors = parse_authors(row[3])
            abstract = parse_abstract(row[4])
            string += f"* [[{title}]({url_abs})]\n"
            string += f"    [[pdf]({url_pdf})]\n"
            string += f"    * Title: {title}\n"
            string += f"    * Year: {year}\n"
            string += f"    * Authors: {authors}\n"
            string += f"    * Abstract: {abstract}\n"

    with open('scripts/parsed_paper_info.md', 'w') as f:
        f.write(string)
    print("Parsed markdown string saved to parsed_paper_info.md.")
