import requests
from bs4 import BeautifulSoup
import markdownify
from urllib.parse import urldefrag, urlparse
import os
import pandas as pd
from tqdm import tqdm
import glob


def get_internal_path(tree, relative_path_type=True, name = ""):
    links = []
    for ele in tree:
        try: link,_ = urldefrag(ele['href'])
        except: continue
        if relative_path_type:
            if ((link not in links) and 
                (not (link.startswith('http'))) and
                (not (link.startswith('/'))) and
                (not (link.startswith('./'))) and
                (not (link.startswith('../'))) and
                len(link)>1 and
                link != "index.html"
               ):
                links.append(link)
        else:
            if name: name+'/'
            if ((link not in links) and 
                ((link.startswith('/docs/'+name)) or (link.startswith('/'+name)))):
                links.append(link)
    return links

def get_html(url, name, relative_path_type = True, cls = ''):
    response = requests.get(url)
    article, tree = None,[]
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        if cls == 'main':
            article = soup.find('main')
        elif cls:
            article = soup.find('div',class_=cls)
        else:
            article = soup.find('article')
        if not article:
            article = soup.find('div',class_='body')
        if not article:
            article = soup.find('body')
        if not article:
            article = soup
        for s in article.select('script'):
            s.extract()
        for s in article.select('aside'):
            s.extract()
        for s in article.select('style'):
            s.extract()
        tree = soup.find_all('a')
    if 'huggingface' in url:
        links = get_internal_path(tree, False, name)
    else:
        links = get_internal_path(tree, relative_path_type)
    return article, links

def get_docs_relative(url,name,cls=''):
    article, main_links = get_html(url,name,cls=cls)
    url = url[:url.rfind('/')]+'/'
    main_page = markdownify.markdownify(str(article), strip=['a'],heading_style="ATX")
    if not os.path.exists('data/'+name): os.makedirs('data/'+name)
    c = 0
    with open('data/'+name+'/index_'+str(c)+'.txt', 'w') as f: f.write(main_page)
    print("Number of files to Collect: ",len(main_links), " Files")
    for link in main_links:
        c+=1
        file_path = 'data/'+name+'/'+(link[:link.rfind('.')].replace('/','_'))+('_'+str(c)+'.txt')
        print(file_path)
        in_url = url+link
        m_url = in_url
        if os.path.exists(file_path):
            continue
        article, links= get_html(in_url,name,cls=cls)
        content = markdownify.markdownify(str(article), strip=['a'],heading_style="ATX")
        in_url = in_url[:in_url.rfind('/')]+'/'
        for lnk in links:
            if (not (lnk in main_links)) and ((in_url+lnk) != m_url):
                article, _ = get_html(in_url+lnk,name,cls=cls)
                content += "\n\n"+markdownify.markdownify(str(article), strip=['a'],heading_style="ATX")
        with open(file_path, 'w') as f: f.write(content)

def get_docs_absolute(url,name,cls=''):
    _, main_links = get_html(url, name, False,cls=cls)
    if not os.path.exists('data/'+name): os.makedirs('data/'+name)
    files = {}
    pth = ''
    if 'huggingface' in url: pth = "/" + name
    for l in main_links:
        f_name = (l.replace('/docs'+pth,'')[1:]).split('/')[0]
        if f_name in files:
                files[f_name].append(l)
        else:
            files[f_name] = [l]
    base_url = urlparse(url)
    base_url = str(base_url.scheme)+"://"+str(base_url.netloc)
    c = 0
    for key in files:
        links = files[key]
        content = ""
        for l in links:
            html, _ = get_html(base_url+l, name, False, cls=cls)
            content += "\n\n"+markdownify.markdownify(str(html), strip=['a'],heading_style="ATX")
        with open('data/'+name+'/'+key+'_'+str(c)+'.txt', 'w') as f: f.write(content)
        c+=1

def remove_duplicate_lines(text_str):
    l = text_str.split("\n\n")
    temp = []
    for x in l:
        if x not in temp:
            temp.append(x)
    return '\n\n'.join(temp)

def clean_files(path):
    paths = []
    for p in glob.glob(path+"/*"): 
        with open(p) as f:
            file_content = f.read()
            file_content = remove_duplicate_lines(file_content)
            with open(p, 'w') as f: 
                f.write(file_content)
            

def combine_files(path):
    paths = []
    for p in glob.glob(path+"/*"): 
        paths.append(p.replace(path+'/',''))
    files = []
    for p in paths:
        f = p.replace('.txt','').split('_')
        f_name = f[0]
        try: f_num = int(f[-1])
        except: continue
        files.append((f_num,f_name,p))
    files.sort(key=lambda x: x[0])
    total_content = ""
    while files:
        i = files[0]
        file_content = ""
        with open(path+'/'+i[2]) as f: file_content += "\n\n" + f.read()
        os.remove(path+'/'+i[2])
        files.remove(i)
        j = 0
        while j < len(files):
            c=files[j]
            if (i[1]==c[1]):
                with open(path+'/'+c[2]) as f: file_content += "\n\n" + f.read()
                os.remove(path+'/'+c[2])
                files.remove(c)
            else:
                j+=1
        with open(path+'/'+i[1]+'.txt', 'w') as f: f.write(file_content)
        total_content += file_content
    with open(path+'/total_content.txt', 'w') as f: f.write(total_content)
    return files

apis_list = pd.read_csv("APIs List.csv")
apis_list.sample()


for i in tqdm(range(0,len(apis_list))):
    api = apis_list.iloc[[i]]
    name = api['Name'][i]
    print(name+" ........ Downloading")
    url = api['Documentation'][i]
    docs_type = api['Docs Type'][i].split('_')
    cls = docs_type[0]
    if cls == 'article': cls = ''
    if len(docs_type)<2: 
        get_docs_relative(url,name,cls)
    else:
        get_docs_absolute(url,name,cls)
    
    print(name+" ........ Cleaning")
    combine_files('data/'+name)
    clean_files('data/'+name)
    
    print(name+" ........ Done")

