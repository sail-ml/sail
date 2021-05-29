from bs4 import BeautifulSoup,Tag

try:
    html = open("_build/html/index.html", "r")
    soup = BeautifulSoup(html.read(),'html.parser')

    soup.find(id='credit').decompose()

    with open("_build/html/index.html", "wb") as f_output:
        f_output.write(soup.prettify("utf-8")) 
except:
    pass

from shutil import copyfile


copyfile("sail-logo16.png", "_build/html/_static/img/favicon-16x16.png")
copyfile("sail-logo32.png", "_build/html/_static/img/favicon-32x32.png")