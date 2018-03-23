'''
Defines class for scraping web data required
for training machine learning models
'''
import scrapy
from IPython import embed

class LiquorSpider(scrapy.Spider):
    name = "liquor_spider"
    start_urls = ['https://www.liquor.com/hub/cocktail-recipes/']

    def parse(self, response):
        href_selector = 'body > div.row.inner-content-width > div.col-xs-6 > a.overlay::attr(href)'
        for href in response.css(href_selector):
            url = response.urljoin(href.extract())
            yield scrapy.Request(url, callback=self.parse_category)

    def parse_category(self, response):
        href_selector = '#mosaic > div > div.mosaic-row.sort-quads a::attr(href)'
        for i, href in enumerate(response.css(href_selector)):
            if i == 0: continue
            url = response.urljoin(href.extract())
            yield scrapy.Request(url, callback=self.parse_description)

    def parse_description(self, response):
        name = response.css('body > div.container.recipe-container.full-recipe-container > div.row.head-row.text-center > div > h1::text').extract()
        ingredients = response.css('div.x-recipe-ingredients div.x-recipe-ingredient *::text').extract()
        descriptions = response.css('body > div.container.recipe-container.full-recipe-container > div.row.image-row > div.col-xs-12.col-sm-7.col-md-8.white-box > div:nth-child(2) > div.row.ingredients-preparation.acid-links > div.col-sm-6.col-xs-12 > div p')
        if not ingredients:
            ingredients = response.css('div.x-recipe-ingredients li[itemprop=ingredients]')
            ingredients = [''.join(x.css('::text').extract()) for x in ingredients]
        if not name or not ingredients or not descriptions: return
        description = ' '.join([x for x in descriptions.css('::text').extract()])
        ingredients = '; '.join([x.strip() for x in ingredients if x.strip()])
        obj = {
            'name': name[0],
            'ingredients': ingredients,
            'description': description
        }
        yield obj
