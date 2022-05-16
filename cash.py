from spotipy import CacheHandler

cashMoney=[]

class CashHandler(CacheHandler):
  def get_cached_token(self):
        if (len(cashMoney)>0):
            return cashMoney.pop()
        else: return ""

  def save_token_to_cache(self,token_info):
      cashMoney.append(token_info)

  pass