from bs4 import BeautifulSoup,Tag

try:
    html = open("_build/html/index.html", "r")
    soup = BeautifulSoup(html.read(),'html.parser')

    soup.find(id='credit').decompose()

    with open("_build/html/index.html", "wb") as f_output:
        f_output.write(soup.prettify("utf-8")) 
except:
    pass