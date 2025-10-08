import base64
import requests
import json

client_id = "381d55411d3447b2bca7de384e49c139"
client_secrets = 'f563d83cad204d67bd9330326d091703'
code = 'AQBtoh_EXQhCYUSynGkRwdMZ7pHp0Tl6B6ZH54tft1ON4bCWpfOOgVFgxZqDIpkoM7FSdBjD72Jx1QorvytFcvXAvghyGJBsJzgzoJFI4qB3HM9wwxgqTXVKeGC6pb6Q_hMWeTQV16ekl6dp4PbE9SOM4JT-C9y48K4dSCQDnmstsXoQogWqi_0Py1EYmalBAOHQjxN5qKSnexLaYf535ivyVRcKmsLGs6bjpRw3B3fsHM8Lj-YdgEWAlw-3rxqmYnmfr-d8tUHsJYk2qb4kPsJO-eVnMdmu_TBRbJsUy22CH3AErofzXTeP-xmr8iGcrJiXTBfYxRRRbxEiaKo9JbdxqzKZDoNDglyfpTD9B0JYAvrx5Bv2MERQhRuKlQ'

REDIRECT_URI = 'http://127.0.0.1:8888/callback'

class SpotifyAPI:
    def __init__(self):
        self.token = self.get_token()
        self.playlist = self.get_all_playlist()

    def get_token(self):
        auth_string = client_id + ':' + client_secrets
        auth_bytes = auth_string.encode('utf-8')
        auht_base64 = str(base64.b64encode(auth_bytes), 'utf-8')
        url = 'https://accounts.spotify.com/api/token'
        headeres = {
            "Authorization": "Basic " + auht_base64,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI
        }
        result = requests.post(url, headers=headeres, data=data)
        json_result = json.loads(result.content)
        print(json_result)
        token = json_result['access_token']
        return token
    
    @staticmethod
    def get_auth_header(token):
        return {"Authorization": "Bearer "+token}
    
    def get_all_playlist(self):
        url = "https://api.spotify.com/v1/me/playlists"
        headers = self.get_auth_header(self.token)
        params = {'limit': 50, 'offset': 0}

        all_playlist = []
        while True:
            result = requests.get(url, headers=headers, params=params)
            data = json.loads(result.content)
            for play in data['items']:
                print(play['name'])
            all_playlist.extend(data['items'])
            if not data.get('next'):
                break
            url = data['next']
            params = None
        return all_playlist
    
    def get_playlist(self, name_playlist):
        playlist_id = None
        for playlist in self.playlist:
            if playlist['name'] == name_playlist:
                playlist_id = playlist['id']
        if playlist_id:
            return playlist_id
        print(f"'{name_playlist}' does not exist or couldnt be found")
        print("Press ENTER")
        input()
        return None

    def get_tracks_fromplaylist(self, playlist_id):
        url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
        headers = self.get_auth_header(self.token)
        params = {'limit': 100, 'offset': 0}
        
        uris = []
        while True:
            result = requests.get(url, headers=headers, params=params)
            data = json.loads(result.content)
            for thing in data['items']:
                print(thing['track']['name'])
            #print(data['tracks']['items'])
            uris.extend(item['track'] for item in data['items'])
            print(len(uris))
            if not data.get('next'):
                break
            url = data['next']
            params = None
        return uris

    def empty_playlist(self, name, id):
        headers = self.get_auth_header(self.token)
        if id:
            headers['Content-Type'] = 'application/json'
            url = f'https://api.spotify.com/v1/playlists/{id}/tracks'
            body = {'uris': []}
            result = requests.put(url, headers=headers, json=body)
            print(result.status_code)
            return id
        else:
            url = 'https://api.spotify.com/v1/me'
            result = requests.get(url, headers=headers)
            user_id = json.loads(result.content)['id']
            headers['Content-Type'] = 'application/json'
            url = f'https://api.spotify.com/v1/users/{user_id}/playlists'
            nie = False
            body = {
                'name': name,
                'public': nie
            }
            result = requests.post(url, headers=headers, json=body)
            print(result.status_code)
            result = json.loads(result.content)
            return result['id']

    def add_songs(self, songs, id_play):
        url = f'https://api.spotify.com/v1/playlists/{id_play}/tracks'
        headers = self.get_auth_header(self.token)
        headers['Content-Type'] = 'application/json'
        if 'uri' in songs:
            body = {
                'uris': [songs['uri']]
            }
            print(songs)
        else:
            print(songs)
            body = {
                'uris': [song['uri'] for song in songs]
            }
        result = requests.post(url, headers=headers, json=body)
        data = json.loads(result.content)
        print(data)

    def create_new_playlists(self, listofsongs, name_playlists, name_new_playlists = ['gym', 'study', 'self', 'commu', 'like']):
        name_new_playlists = name_new_playlists[-len(listofsongs):]
        old_play_id = [self.get_playlist(name) for name in name_playlists]
        old_play = [self.get_tracks_fromplaylist(play_id) for play_id in old_play_id]
        print(sum([len(old) for old in old_play]))
        songs = [song for period in old_play for song in period]
        #print(songs)
        print(len(songs))
        input()
        songs_theme = []
        for liste in listofsongs:
            print(liste)
            songs_theme.append([songs[i] for i in liste])
            print(songs[-1])
            input()
        new_play_id = [self.get_playlist(name) for name in name_new_playlists]
        new_play_id = [self.empty_playlist(name, id) for name, id in zip(name_new_playlists,new_play_id)]
        [self.add_songs(song, id_play) for song, id_play in zip(songs_theme, new_play_id)]
