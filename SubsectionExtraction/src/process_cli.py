from src.preprocessing.html_parser import SANHtmlParser


def add_border(html_block):
    # return '<div style="border: ridge red; border-width: 2px 2px 2px 4px">' + html_block + '</div>'
    return '<div style="background-color: lightblue; border: ridge red; border-width: 1px 1px 1px 1px">' + html_block + '</div>'


if __name__ == '__main__':
    file_name = '0001193125-20-053557-sec_bs.htm'
    file_path_template = '/home/vagrant/data/shared/PoC-Block/{}'

    with open(file_path_template.format(file_name), "r") as f_in:
        doc = f_in.read()

    # Cut the original html into blocks
    san_html_parser = SANHtmlParser()

    san_html_parser.feed(doc)

    blocks = san_html_parser.to_blocks()

    # Rebind the html
    parsed_html = '<br/>'.join([add_border(block.html) for block in blocks])
    with open(file_path_template.format('blocked11_' + file_name), 'w') as f_out:
        f_out.write(parsed_html)
