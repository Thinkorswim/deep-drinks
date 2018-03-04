'''
Defines class for scraping web data required
for training machine learning models
'''
import scrapy
import json
from IPython import embed

class SeriousEatsSpider(scrapy.Spider):
    name = "seriouseats_spider"
    start_urls = [
    'https://www.seriouseats.com/topics?index=recipes&count=1000&before=0&tag=c|recipes|topics|meal|drinks|cocktails|brandy-and-cognac&only=id,permalink,title,short_title,kicker,blog_id&jsonp=jQuery2240892462437412537_1520087407298&_=1520087407299',
    'https://www.seriouseats.com/topics?index=recipes&count=1000&before=0&tag=c|recipes|topics|meal|drinks|cocktails|champagne-cocktails&only=id,permalink,title,short_title,kicker,blog_id&jsonp=jQuery2240892462437412537_1520087407298&_=1520087407299',
    'https://www.seriouseats.com/topics?index=recipes&count=1000&before=0&tag=c|recipes|topics|meal|drinks|cocktails|gin&only=id,permalink,title,short_title,kicker,blog_id&jsonp=jQuery2240892462437412537_1520087407298&_=1520087407299',
    'https://www.seriouseats.com/topics?index=recipes&count=1000&before=0&tag=c|recipes|topics|meal|drinks|cocktails|liqueur-and-fortified-wines&only=id,permalink,title,short_title,kicker,blog_id&jsonp=jQuery2240892462437412537_1520087407298&_=1520087407299',
    'https://www.seriouseats.com/topics?index=recipes&count=1000&before=0&tag=c|recipes|topics|meal|drinks|cocktails|mezcal&only=id,permalink,title,short_title,kicker,blog_id&jsonp=jQuery2240892462437412537_1520087407298&_=1520087407299',
    'https://www.seriouseats.com/topics?index=recipes&count=1000&before=0&tag=c|recipes|topics|meal|drinks|cocktails|rum&only=id,permalink,title,short_title,kicker,blog_id&jsonp=jQuery2240892462437412537_1520087407298&_=1520087407299',
    'https://www.seriouseats.com/topics?index=recipes&count=1000&before=0&tag=c|recipes|topics|meal|drinks|cocktails|sangria&only=id,permalink,title,short_title,kicker,blog_id&jsonp=jQuery2240892462437412537_1520087407298&_=1520087407299',
    'https://www.seriouseats.com/topics?index=recipes&count=1000&before=0&tag=c|recipes|topics|meal|drinks|cocktails|tequila&only=id,permalink,title,short_title,kicker,blog_id&jsonp=jQuery2240892462437412537_1520087407298&_=1520087407299',
    'https://www.seriouseats.com/topics?index=recipes&count=1000&before=0&tag=c|recipes|topics|meal|drinks|cocktails|vodka&only=id,permalink,title,short_title,kicker,blog_id&jsonp=jQuery2240892462437412537_1520087407298&_=1520087407299',
    'https://www.seriouseats.com/topics?index=recipes&count=1000&before=0&tag=c|recipes|topics|meal|drinks|cocktails|whiskey&only=id,permalink,title,short_title,kicker,blog_id&jsonp=jQuery2240892462437412537_1520087407298&_=1520087407299'
    ]

    def parse(self, response):
        raw_data = response.css('body > p').extract()[0]
        raw_data = ''.join((''.join(raw_data.split('(')[1:])).split(')')[:-1])
        data = json.loads(raw_data)
        for item in data['entries']:
            url = item['permalink']
            yield scrapy.Request(url, callback=self.parse_description)


    def parse_description(self, response):
        name = response.css('h1.title.recipe-title::text').extract()
        ingredients = response.css('div.recipe-ingredients ul > li::text').extract()
        description = response.css('div.recipe-procedures div.recipe-procedure-text > p:nth-of-type(2)::text').extract()
        if not name or not ingredients or not description: return
        ingredients = '; '.join([i.strip() for i in ingredients])
        description = ' '.join([d.strip() for d in description])
        obj = {
            'name': name[0],
            'ingredients': ingredients,
            'description': description
        }
        yield obj
