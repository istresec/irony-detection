import re

if __name__ == '__main__':
    regex = re.compile(r'(\.{2,5})([^.]|$)|(\.{3})')
    text = regex.sub(r' ... \2', "Mary.... is..... .........very.....")
    print(text)
    text = regex.sub(r' ... \2', ".... Wait what does.......what does this have to do with anything...")
    print(text)
    ellipsis_count = len(regex.findall("Mary.... is..... .........very....."))
    print(ellipsis_count)