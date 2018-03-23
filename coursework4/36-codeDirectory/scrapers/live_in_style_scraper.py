'''
Defines class for scraping web data required
for training machine learning models
'''
import scrapy
from IPython import embed

class LiveInStyleSpider(scrapy.Spider):
    name = "liveinstyle_spider"
    start_urls = [
        'https://www.liveinstyle.com/cocktails/gin-cocktails',
        'https://www.liveinstyle.com/cocktails/scotch-and-whisky-cocktails',
        'https://www.liveinstyle.com/cocktails/rum-cocktails',
        'https://www.liveinstyle.com/cocktails/vodka-cocktails',
        'https://www.liveinstyle.com/cocktails/wine-cocktails'
    ]

    def parse(self, response):
        href_selector = '#displayCocktail > li > div > a::attr(href)'
        for href in response.css(href_selector):
            url = response.urljoin(href.extract())
            yield scrapy.Request(url, callback=self.parse_description)
        next_selector = 'li.pager-next > a::attr(href)'
        next_url = response.css(next_selector).extract_first()
        if next_url:
            next_url = response.urljoin(next_url)
            yield scrapy.Request(next_url, callback=self.parse)


    def parse_description(self, response):
        remove_unicode = lambda x: x.encode('ascii', errors='ignore').decode().strip()
        name = response.css('h1[itemprop=name]::text').extract()
        ingredients = response.css('ul[itemprop=recipeIngredient] *::text').extract()
        description = response.css('ul[itemprop=recipeInstructions] *::text').extract()
        if not ingredients:
            ingredients = response.css('div[itemprop=recipeIngredient] *::text').extract()
        if not description:
            description = response.css('div[itemprop=recipeInstructions] > p:nth-of-type(2) *::text').extract()
        if not name or not ingredients or not description: return
        ingredients = '; '.join([remove_unicode(i) for i in ingredients if remove_unicode(i)])
        description = ' '.join([remove_unicode(d) for d in description if remove_unicode(d)])
        obj = {
            'name': name[0],
            'ingredients': ingredients,
            'description': description
        }
        yield obj
