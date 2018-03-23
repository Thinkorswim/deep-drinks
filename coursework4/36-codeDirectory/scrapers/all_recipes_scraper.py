'''
Defines class for scraping web data required
for training machine learning models
'''
import scrapy
from IPython import embed

class AllRecipesSpider(scrapy.Spider):
    name = "allrecipes_spider"
    start_urls = [
        'https://www.allrecipes.com/recipes/133/drinks/cocktails/?page=1'
    ]

    def parse(self, response):
        title = response.css('title::text').extract()[0]
        if title == 'Allrecipes - File Not Found':
            return
        href_selector = 'article.fixed-recipe-card > a[href]::attr(href)'
        for href in response.css(href_selector):
            url = response.urljoin(href.extract())
            yield scrapy.Request(url, callback=self.parse_description)
        current_page = int(response.url.split('?page=')[1])
        next_url = response.url.split('?')[0] + '?page=' + str(current_page + 1)
        yield scrapy.Request(next_url, callback=self.parse)

    def parse_description(self, response):
        name = response.css('h1[itemprop=name]::text').extract()
        ingredients = response.css('ul[id^=lst_ingredients] *[itemprop=ingredients]::text').extract()
        description = response.css('*[itemprop=recipeInstructions] *::text').extract()
        if not name or not ingredients or not description: return
        ingredients = '; '.join([i.strip() for i in ingredients if i]).strip()
        description = ' '.join([d.strip() for d in description if d]).strip()
        obj = {
            'name': name[0],
            'ingredients': ingredients,
            'description': description
        }
        yield obj
