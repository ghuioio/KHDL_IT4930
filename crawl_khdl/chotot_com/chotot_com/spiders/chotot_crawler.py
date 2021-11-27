import scrapy
from ..items import ChototItem
class ChototSpider(scrapy.Spider):
    name = "chotot"
    page_number = 2
    base_url = "https://xe.chotot.com/mua-ban-oto"
    start_urls = ["https://xe.chotot.com/mua-ban-oto?page=1"]

    def parse(self, response):
        all_ads = response.xpath('//li[contains(@class, "AdItem")]')

        for ad in all_ads:
            ad_url = ad.xpath('.//a[contains(@class, "AdItem")]/@href').extract_first()

            ad_url = self.base_url + ad_url
            yield scrapy.Request(url=ad_url, callback=self.parse_ad)
    
    def parse_ad(self, response):
        item = ChototItem()
        item['gia'] = response.xpath('//span[contains(@itemprop, "price")]/text()').extract_first().replace(' đ', '')
        item['hang'] = response.xpath('//span[contains(@itemprop, "carbrand")]/text()').extract_first()
        item['dong_xe'] = response.xpath('//span[contains(@itemprop, "carmodel")]/text()').extract_first()
        item['nam_sx'] = response.xpath('//span[contains(@itemprop, "mfdate")]/text()').extract_first()
        item['tinh_trang'] = response.xpath('//span[contains(@itemprop, "condition")]/text()').extract_first()
        item['nhien_lieu'] = response.xpath('//span[contains(@itemprop, "fuel")]/text()').extract_first()
        item['kieu_dang'] = response.xpath('//span[contains(@itemprop, "cartype")]/text()').extract_first()
        item['km_da_di'] = response.xpath('//span[contains(@itemprop, "mileage")]/text()').extract_first()
        item['hop_so'] = response.xpath('//span[contains(@itemprop, "gearbox")]/text()').extract_first()
        item['xuat_xu'] = response.xpath('//span[contains(@itemprop, "origin")]/text()').extract_first()
        item['so_cho_ngoi'] = response.xpath('//span[contains(@itemprop, "carseat")]/text()').extract_first()

        next_page = 'https://xe.chotot.com/mua-ban-oto?page=' + str(ChototSpider.page_number)
        if ChototSpider.page_number <= 10000:
            ChototSpider.page_number += 1
            yield response.follow(url=next_page, callback=self.parse)
        yield item 