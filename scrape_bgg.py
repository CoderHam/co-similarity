from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
from tqdm import tqdm

class BoardGamesGeekScraper(object):
    def __init__(self) -> None:
        self.PAGE_LIMIT = 20
        self.PREFIX_URL = "https://boardgamegeek.com"
        self.PAGE_URL = "/browse/boardgame/page/"
        self.counter = 0

    def fill_details(self, records):
        for idx, row in records.iterrows():
            page = requests.get(row['url'])
            soup = BeautifulSoup(page.content, "html.parser")
            item_payload = soup.find_all("script")[2].get_text()
            start_idx = item_payload.find("GEEK.geekitemPreload = ") + len("GEEK.geekitemPreload = ")
            end_idx = item_payload.find("};", start_idx) + 1
            item_json = json.loads(item_payload[start_idx:end_idx])['item']

            # get rank information and rating
            for rankinfo in item_json['rankinfo']:
                if rankinfo['veryshortprettyname'] == 'Overall':
                    records.at[idx,'rank'] = int(rankinfo['rank'])
                    records.at[idx,'rating'] = float(rankinfo['baverage'])

            # get description
            records.at[idx,'description'] = BeautifulSoup(item_json['description'], "html.parser").get_text()

            # get thumbnail
            records.at[idx,'thumbnail'] = item_json['images']['previewthumb']
        
        return records

    def get_info_for_page(self, page_num: int):
        page = requests.get(self.PREFIX_URL + self.PAGE_URL + str(page_num))
        soup = BeautifulSoup(page.content, "html.parser")
        titles = soup.find_all("a", {"class": "primary"})
        records = None

        # get all games in this page
        for title in titles:
            name = title.get_text()
            url = self.PREFIX_URL + title['href']
            row = {'uid': self.counter, 'name': name, 'url': url, 'rating': 0.0, 'rank': -1, 'description': '', 'thumbnail': ''}
            df = pd.DataFrame(row, index=[self.counter])
            if records is None:
                records = df
            else:
                records = pd.concat([records, df])
            self.counter += 1

        return self.fill_details(records)

    def scrape_info(self) -> None:
        all_records = []
        progress_bar = tqdm(range(1, self.PAGE_LIMIT + 1))
        for page_num in progress_bar:
            progress_bar.set_description("Extracting records from page %s" % page_num)
            records = self.get_info_for_page(page_num)
            if records is None:
                all_records = [records]
            else:
                all_records.append(records)
                
        merged_records = pd.concat(all_records)

        # Save records to csv file
        merged_records.to_csv("bgg_2000.csv", index=False)

BGG = BoardGamesGeekScraper()
BGG.scrape_info()
print("saved to file")
