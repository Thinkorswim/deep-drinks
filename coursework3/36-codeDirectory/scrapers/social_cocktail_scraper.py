'''
Defines class for scraping web data required
for training machine learning models
'''
import scrapy
from IPython import embed

class SocialCoctailSpider(scrapy.Spider):
    name = "socialcoctail_spider"
    start_urls = ['https://www.socialandcocktail.co.uk/cocktail-recipes/']

    def parse(self, response):
        href_selector = '#content-home > section > div.recipe_summary > a::attr(href)'
        for href in response.css(href_selector):
            url = href.extract()
            yield scrapy.Request(url, callback=self.parse_description)
        next_selector = '#content-home > section > div.paginate.pjax > div > a.nextpostslink::attr(href)'
        next_url = response.css(next_selector).extract_first()
        if next_url:
            yield scrapy.Request(next_url, callback=self.parse)


    def parse_description(self, response):
        name = response.css('#content-home > section > div.single_header > h2::text').extract()
        description = response.css('#content-to-load > div.recipe-content > p:nth-child(4)::text').extract()
        if not name or not description: return
        obj = {
            'name': name[0],
            'description': description[0]
        }
        yield obj
