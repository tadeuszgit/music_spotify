import urllib.parse as up

CLIEN_ID = "381d55411d3447b2bca7de384e49c139"
REDIRECT_URI = 'http://127.0.0.1:8888/callback'
SCOPE = 'playlist-modify-private playlist-modify-public playlist-read-private user-read-playback-position user-top-read user-library-modify user-library-read'

state = 123456790123456
params = {
    'client_id': CLIEN_ID,
    'response_type': 'code',
    'redirect_uri': REDIRECT_URI,
    'scope': SCOPE,
    'state': state
}

url = 'https://accounts.spotify.com/authorize?' + up.urlencode(params)
print('GRANT THE ACCESS')
print(url)