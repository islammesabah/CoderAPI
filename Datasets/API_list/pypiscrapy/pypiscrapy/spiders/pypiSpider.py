# To tun the script: scrapy crawl pypi -O pypi_2.csv

import scrapy
import pandas as pd

class PypiSpider(scrapy.Spider):
    name = 'pypi'
    # allowed_domains = ['pypi.org']
    # start_urls = ['https://pypi.org/classifiers/']

    def start_requests(self):
        packages_url = '../package-dataframe.csv'
        df = pd.read_csv(packages_url)
        data = [x.strip() for x in df['package_name']]
        for row in data:
            print(row)
            yield scrapy.Request(
                url=f'https://pypi.org/project/{row}/',
                callback=self.parse,
            )

    def parse(self, response):
        name = response.css('h1.package-header__name::text').get().strip().split(' ')
        install = response.css('span[id=pip-command]::text').get()
        release_date = response.css('time::attr(datetime)').get().split('T')[0]
        # discription =  response.css('div[id="description"] div.project-description').get()
        old_version = response.css('div[class="release release--oldest"] p.release__version::text').get().strip(),
        old_release_date = response.css('div[class="release release--oldest"] time::attr(datetime)').get().split('T')[0]
        
        yield {
            'name': name[0],
            'version': name[1],
            'install_command': install,
            'release_date': release_date,
            'release_year': release_date.split('-')[0],
            'old_version': old_version[0],
            'old_release_date': old_release_date,
            'old_release_year': old_release_date.split('-')[0],
            'url': response.request.url
            # 'discription': discription
        }
      
        # for cls in response.selector.xpath('//li[@data-controller="clipboard"]//a/@href'):
            # yield response.follow(self.allowed_domains[0]+cls, callback=self.parse_cls)

        pass