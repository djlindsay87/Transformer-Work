import datetime
import time
import spotipy
import random
import regex as re
import operator
import requests
from bs4 import BeautifulSoup as bs
from functools import reduce
import unicodedata as ucd
import unidecode as udc
from torch.utils.data import random_split
from spotipy.oauth2 import SpotifyClientCredentials
from concurrent.futures import ThreadPoolExecutor

# import cash # I guess only include if you're on replit or something?

class TheBeast(tuple):
    def __init__(self, artistURI:str='spotify:artist:3WrFJ7ztbogyGnTHbHJFl2', nAlbums:int=None, nSongs:int=None, yRange:tuple=None, tokType:str='word'):
        self.t0=time.time()
        print("Bootin' the Beast...")
        self.tokType=tokType.lower()
        self.textTokDict:dict
        self.tokTextDict:dict
        self.testToks:list
        self.trainToks:list
        self.valToks:list
        self.artistURI=artistURI
        self.nAlbums=nAlbums
        self.nSongs=nSongs
        self.yRange=yRange
        
        if self.yRange:
            self.yStart=min(yRange)
            self.yEnd=max(yRange) 
        
        print("Initializing spotify connection...")
        __cid = 'bac7c5b352224d3ead9934fe4554ac1c'
        __secret =  '7b226000998a410789eb7be62fda9a02'# use your own if you got one
        

        __ccm = SpotifyClientCredentials(client_id=__cid, client_secret=__secret)
        self.sp = spotipy.Spotify(client_credentials_manager = __ccm, requests_timeout=60)


        print('Trying artist token... ',end='')
        self.artistName=self.sp.artist(artistURI)['name']
        print('Successful.')
        self.songList=self.__catalogueCollector()
        self.sesh=requests.Session()
        executor=ThreadPoolExecutor(max_workers=5)
        self.lyricList=list(filter(lambda s:s!="<!--INVALID URL-->", executor.map(self.__geniusScraper,self.songList)))   
        self.nLyrics=len(self.lyricList)
        tlen= 3*self.nLyrics//5
        vlen=(self.nLyrics-tlen)//2
        tstlen=self.nLyrics-tlen-vlen    
        t3=time.time()

        print(f"{self.nLyrics} valid song lyrics scraped in {int((t3-self.t0)//60)} min and {(t3-self.t0)%60:.3f}s.")
        self.trainLyrics, self.testLyrics, self.valLyrics = random_split(self.lyricList,[tlen,tstlen,vlen])
        print(f"Split to {tlen} train songs, {tstlen} test songs, {vlen} val songs... ",end='')
        self.tokenTuple=self.__tokenify(); print("Tokenized.")
        return

    
    def __len__(self):
        return (len(self.tokTextDict))
        
    def __str__(self):
        return f"Train text tokens: {len(self.trainToks)},\nTest text tokens: {len(self.testToks)},\nValidation text tokens: {len(self.valToks)}"
        
    def __getitem__(self, idx):
        return self.tokenTuple[idx]
        
    def decode(self, tokenList:list=None):
        if not tokenList: return tuple([*map(self.tokTextDict.get, tList)] for tList in self.tokenTuple)
        else: return [*map(self.tokTextDict.get, tokenList)]
        
    def encode(self, text:str):
        textList=self.__splitUp(text)
        return [*map(lambda tok: self.textTokDict.get(tok) if tok in self.textTokDict else 0, textList)]
        
    def __albumRipper(self, albumURI:str)->list:
        chunk=self.sp.album_tracks(albumURI)
        songInfo=chunk['items']
        while chunk['next']:
            chunk=self.sp.next(chunk)
            songInfo.extend(chunk['items'])
        return [song['name'] for song in songInfo]
        
    def __ranger(self, infoList:list)->list:
        infoList = [album for album in infoList if datetime.datetime.strptime(album['release_date'],'%Y-%m-%d').year in range(self.yStart,self.yEnd+1)]
        print(f"ranged to {len(infoList)} albums between years {self.yStart} and {self.yEnd}, ",end=''); return infoList
    
    def __listerReducer(self, albumDict:dict) -> list:
        songList=reduce(operator.concat, [albumDict[album]['Song List'] for album in albumDict])
        songList = list(set([re.sub(r'(.+)\s\-\s.+',r'\1',song) for song in songList]))
        print(f"{len(songList)} UNIQUE songs pulled",end='')
        if (self.nSongs in range(1, len(songList)+1)):
            songList=random.sample(songList,self.nSongs)
            print(f", {self.nSongs} sampled",end='')
        return songList
    
    def __catalogueCollector(self) -> dict:
        chunk=self.sp.artist_albums(self.artistURI,album_type='album')
        albumInfo=chunk['items']
        while chunk['next']:
            chunk=self.sp.next(chunk)
            albumInfo.extend(chunk['items'])
        t2=time.time(); print(f"{t2-self.t0:.3f}s to find {len(albumInfo)} albums, ",end='')
        if self.yRange: albumInfo=self.__ranger(albumInfo)
        if (self.nAlbums in range(1,len(albumInfo)+1)):
            albumInfo=random.sample(albumInfo,self.nAlbums)
            print(f"{len(albumInfo)} albums sampled, ",end='')
    
        ### Quin's contribution?
        for album in albumInfo:
            date = album['release_date']
            try: 
                datetime.datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                album['release_date'] = '1999-01-01'
                
        albumDict={album['name']:{'Song List':self.__albumRipper(album['uri']),
                                  'URI':album['uri'],
                                  'Year':datetime.datetime.strptime(album['release_date'], '%Y-%m-%d').year} for album in albumInfo}
        songList=self.__listerReducer(albumDict); t2=time.time()
        print(f'...\n{len(songList)} songs acquired in {t2-self.t0:.3f}s.')
        return songList
    
    def __normalfy(self, string: str)->str:
        return udc.unidecode(ucd.normalize('NFKD', string))
    
    def __geniusify(self, string: str) -> str:
        return re.sub(r'(-)+',r'\1',re.sub(r'[^\w-]','',re.sub(r'[\s\/\\$\_]+','-',self.__normalfy(string))))
        
    def __geniusScraper(self, trackName: str) -> str:
        geniusName=f"{self.__geniusify(self.artistName)}-{self.__geniusify(trackName)}"
        url=f"https://genius.com/{geniusName}-lyrics"
        buffer=""
        page=self.sesh.get(url);
        if not page:
            buffer+=f"INVALID URL -- {geniusName}"
            return "<!--INVALID URL-->"
        else: buffer+=f"{url}"
        soup=bs(page.text, 'html.parser')
        for tag in soup.find_all('br'): tag.replaceWith('\n'+tag.text)
        divlist=soup.find_all("div", class_="Lyrics__Container-sc-1ynbvzw-6 jYfhrf")
        lyrics='<sos>'+"".join([p.get_text() for p in divlist])+'<eos>'
        print(buffer)
        
        return self.__normalfy(lyrics)
    
    def __splitUp(self, text:str)->list:
        Brackets = r"[\[\<].*?[\]\>]"; Words = r"\b[^\s\<\[]+\b"; Acronyms=r"(?:[A-Za-z]\.){2,}[a-zA-Z]?"; Lines = r"\S.*\n{1,}"
        if self.tokType=='word':    return re.findall(f"{Brackets}|{Acronyms}|{Words}|[\)\(,.!?\n]",text.lower())
        elif self.tokType=='line':    return re.findall(f"{Brackets}|{Lines}",text)
        elif self.tokType=='char':   return re.findall(f"{Brackets}|[\s\S]",text)
        else:
            print('INVALID TOKTYPE... SELECT "word" "char" OR "line"')
            return []
    
    def __tokenify(self)->tuple:
        trainText="".join([i for i in self.trainLyrics]) #full text
        testText="".join([i for i in self.testLyrics]) #full text
        valText="".join([i for i in self.valLyrics]) #full text
        self.trainToks=self.__splitUp(trainText) #split up according to char/word/line
        self.testToks=self.__splitUp(testText) #split up according to char/word/line
        self.valToks=self.__splitUp(valText) #split up according to char/word/line
        fullSet=set(self.trainToks+self.testToks+self.valToks) # sum of split tokens condensed to set, so we have unique tokens for dict!
        self.textTokDict={**{'<unk>':0},**{tex:tok for tok, tex in enumerate(fullSet,1)}}
        self.tokTextDict={**{0:'<unk>'},**{tok:tex for tok, tex in enumerate(fullSet,1)}}
        trnTokens=[*map(self.textTokDict.get, self.trainToks)]
        tstTokens=[*map(self.textTokDict.get, self.testToks)]
        valTokens=[*map(self.textTokDict.get, self.valToks)]
        return trnTokens, tstTokens, valTokens
        
    def __call__(self):
        return self.tokenTuple