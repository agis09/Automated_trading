import json
import requests
import time
import traceback
import pandas as pd

# 約定情報
class Execution(object):
    def __init__(self, jsonobj=None):
        if jsonobj:
            self.id = jsonobj['id']
            self.side = jsonobj['side']
            self.price = jsonobj['price']
            self.size = jsonobj['size']
            self.timestamp = jsonobj['exec_date']
    def to_csv_line(self):
        line = '{},{},{},{},{}\n'
        line = line.format(self.id, self.side, self.price,
                           self.size, self.timestamp)
        return line


# bitFlyerのPublic APIを触るためのクラス
class PublicApi(object):

    API_URI = 'https://api.bitflyer.jp/v1/'

    def __init__(self, currency_pair):
        self.currency_pair = currency_pair

    def __get_request(self, endpoint, params=None):
        response = requests.get(endpoint, params=params)
        if response.status_code < 200 or response.status_code >= 300:
            raise Exception('Bad status code {}. Url={}. Body={}'.format(
                response.status_code,
                response.url,
                response.text))
        return response

    def get_executions(self, count=500, before=0, after=0):
        endpoint = PublicApi.API_URI + 'getexecutions'
        params = {
            'product_code': self.currency_pair,
            'count': count,
            'before': before
        }

        if after > 0: params['after'] = after
        response = self.__get_request(endpoint, params=params)
        results = response.json()
        executions = [Execution(x) for x in results]
        return executions

    def save_all_executions(self, csv_file, count=10000):
        """ Get all execution history from public api """
        with open(csv_file, 'w', encoding='utf-8') as f:
            latest = self.get_executions(count=1)
            before = latest[0].id + 1
            saved = 0

            f.write('id,side,price,size,timestamp\n')
            while before > 1:
                try:
                    executions = self.get_executions(before=before)
                    before = executions[-1].id
                    lines = [x.to_csv_line() for x in executions]
                    f.writelines(lines)
                    saved += len(executions)
                    time.sleep(0.12)
                except:
                    print('Failed to process executions. before={}'.format(before))
                    print(traceback.format_exc())

                if count and saved >= count:
                    break


public = PublicApi('BTC_JPY')
public.save_all_executions('./executions.csv', count=1000000)

csv_file = './executions.csv'
df = pd.read_csv(csv_file, parse_dates=['timestamp'])

df['timestamp'] = df['timestamp'].map(lambda x: x.replace(microsecond=0))


# 1分毎の安値と高値(min, max)、始値と終値(first, last)、および取引量(size) を集計
summary = df[['timestamp', 'price']].groupby(['timestamp']).min().rename(columns={'price': 'min'})
summary = summary.merge(
    df[['timestamp', 'price']].groupby(['timestamp']).max().rename(columns={'price': 'max'}),
    left_index=True, right_index=True)
summary = summary.merge(
    df[['timestamp', 'price']].groupby(['timestamp']).last().rename(columns={'price': 'first'}),
    left_index=True, right_index=True)
summary = summary.merge(
    df[['timestamp', 'price']].groupby(['timestamp']).first().rename(columns={'price': 'last'}),
    left_index=True, right_index=True)
summary = summary.merge(
    df[['timestamp', 'size']].groupby(['timestamp']).sum(),
    left_index=True, right_index=True)

summary.to_csv("min_trade.csv")