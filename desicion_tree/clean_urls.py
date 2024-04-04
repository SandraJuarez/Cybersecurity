import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore
from urllib.parse import urlparse
from tld import get_tld, is_tld
from urllib.parse import urlparse
from tld import get_tld
import os.path






def get_info_from_url(url):
    urldata = pd.read_csv(url)

    #Vamos a obtener las etiquetas
    urldata = pd.read_csv(url)

    # Definir un diccionario que mapee cada etiqueta a un número
    etiqueta_a_numero = {"benign": 0, "defacement": 1, "phishing": 2, "malware": 3}

    # Crear una nueva columna llamada "resultado" y asignar números según la etiqueta
    urldata['resultado'] = urldata['type'].map(etiqueta_a_numero)


    urldata['url_length'] = urldata['url'].apply(lambda i: len(str(i)))
    urldata['hostname_length'] = urldata['url'].apply(lambda i: len(urlparse(i).netloc))
    urldata['path_length'] = urldata['url'].apply(lambda i: len(urlparse(i).path))
    def fd_length(url):
        urlpath= urlparse(url).path
        try:
            return len(urlpath.split('/')[1])
        except:
            return 0

    urldata['fd_length'] = urldata['url'].apply(lambda i: fd_length(i))
    urldata['tld'] = urldata['url'].apply(lambda i: get_tld(i,fail_silently=True))
    def tld_length(tld):
        try:
            return len(tld)
        except:
            return -1

    urldata['tld_length'] = urldata['tld'].apply(lambda i: tld_length(i))
    urldata = urldata.drop("tld",1)


    urldata['count-'] = urldata['url'].apply(lambda i: i.count('-'))
    urldata['count@'] = urldata['url'].apply(lambda i: i.count('@'))
    urldata['count?'] = urldata['url'].apply(lambda i: i.count('?'))
    urldata['count%'] = urldata['url'].apply(lambda i: i.count('%'))
    urldata['count.'] = urldata['url'].apply(lambda i: i.count('.'))
    urldata['count='] = urldata['url'].apply(lambda i: i.count('='))
    urldata['count-http'] = urldata['url'].apply(lambda i : i.count('http'))
    urldata['count-https'] = urldata['url'].apply(lambda i : i.count('https'))
    urldata['count-www'] = urldata['url'].apply(lambda i: i.count('www'))
    def digit_count(url):
        digits = 0
        for i in url:
            if i.isnumeric():
                digits = digits + 1
        return digits
    urldata['count-digits']= urldata['url'].apply(lambda i: digit_count(i))
    def letter_count(url):
        letters = 0
        for i in url:
            if i.isalpha():
                letters = letters + 1
        return letters
    urldata['count-letters']= urldata['url'].apply(lambda i: letter_count(i))
    def no_of_dir(url):
        urldir = urlparse(url).path
        return urldir.count('/')
    urldata['count_dir'] = urldata['url'].apply(lambda i: no_of_dir(i))



    import re
    #Use of IP or not in domain
    def having_ip_address(url):
        match = re.search(
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
            '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
        if match:
            # print match.group()
            return -1
        else:
            # print 'No matching pattern found'
            return 1
    urldata['use_of_ip'] = urldata['url'].apply(lambda i: having_ip_address(i))
    def having_ip_address(url):
        match = re.search(
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
            '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
        if match:
            # print match.group()
            return -1
        else:
            # print 'No matching pattern found'
            return 1
    urldata['use_of_ip'] = urldata['url'].apply(lambda i: having_ip_address(i))
    def shortening_service(url):
        match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                        'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                        'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                        'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                        'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                        'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                        'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                        'tr\.im|link\.zip\.net',
                        url)
        if match:
            return -1
        else:
            return 1
    urldata['short_url'] = urldata['url'].apply(lambda i: shortening_service(i))


    #Predictor Variables
    x = urldata[['hostname_length',
        'path_length', 'fd_length', 'tld_length', 'count-', 'count@', 'count?',
        'count%', 'count.', 'count=', 'count-http','count-https', 'count-www', 'count-digits',
        'count-letters', 'count_dir', 'use_of_ip']]

    #Target Variable
    y = urldata['resultado']

    #save the data
    #x.to_csv('x.csv', index=False)
    #y.to_csv('y.csv', index=False)

    #return the data and the target 
    return x,y



if __name__ == "__main__":
    #url="C:/Users/flaco/OneDrive/Escritorio/hackaton_2024/Cybersecurity/data/malicious_phish.csv"
    x,y=get_info_from_url(url)




