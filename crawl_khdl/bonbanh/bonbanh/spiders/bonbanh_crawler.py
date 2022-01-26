import scrapy
from ..items import BonbanhItem
class BonbanhSpider(scrapy.Spider):
    name = 'bonbanh'
    page_number = 1
    base_url = "https://bonbanh.com/"
    start_urls = ["https://bonbanh.com/oto/page,2"]
    def parse(self, response):
        all_ads = response.xpath('//li[contains(@class, "car-item")]')
        for ad in all_ads:
            ad_url = ad.xpath('.//a[contains(@itemprop, "url")]/@href').extract_first()
            ad_url = self.base_url + ad_url
            yield scrapy.Request(ad_url, callback=self.parse_ad)

    def parse_ad(self, response):
        item = BonbanhItem()
        item['gia'] = response.xpath('//h1/text()').extract_first()
        info = response.xpath('//span[@class = "inp"]')
        item['xuat_xu'] = info[0].xpath('.//text()').extract_first()
        item['tinh_trang'] = info[1].xpath('.//text()').extract_first()
        item['dong_xe'] = info[2].xpath('.//text()').extract_first()
        item['km_da_di'] = info[3].xpath('.//text()').extract_first()
        item['mau_xe'] = info[4].xpath('.//text()').extract_first()
        item['so_cua'] = info[6].xpath('.//text()').extract_first()
        item['so_cho_ngoi'] = info[7].xpath('.//text()').extract_first()
        item['nhien_lieu'] = info[8].xpath('.//text()').extract_first()
        item['hop_so'] = info[10].xpath('.//text()').extract_first()

        next_page = 'https://bonbanh.com/oto/page,' + str(BonbanhSpider.page_number)
        if BonbanhSpider.page_number <= 5000:
            BonbanhSpider.page_number += 1
            yield response.follow(next_page, callback=self.parse)
        yield item