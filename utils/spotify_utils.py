import spotipy
import spotipy.util

#Initialize spotify for API calls
def initialize_spotify():
    token = spotipy.util.prompt_for_user_token('', scope='playlist-read-private',
                                       client_id='',
                                       client_secret='',
                                       redirect_uri='http://localhost/')
    print("token set")
    spotify = spotipy.Spotify(auth=token)
    print("spotify initialized")
    return spotify


#Returns list of spotify album items for given artist name
def get_artist_albums(spotify, artist_name):
    results = spotify.search(q='artist:'+artist_name, type='artist')
    artist_id = results['artists']['items'][0]['external_urls']['spotify'][-22:]
    artist_albums_items = spotify.artist_albums(artist_id)['items']
    return artist_albums_items

#Returns Dictionary containing audio_analysis for track_id
def get_audio_analysis_by_track_id(spotify,track_id):
    track_analysis=spotify.audio_analysis(track_id)
    return track_analysis

#Returns list of Dictionaries containing segments for track id
def get_segments_by_track_id(spotify,track_id):
  track_analysis = spotify.audio_analysis(track_id)
  segments=track_analysis['segments']
  return segments

#Returns list of songs. song is list of segment Dictionaries
def get_segments_for_album(spotify,album_id):
  album_tracks = spotify.album_tracks(album_id)['items']
  num_tracks_i=album_tracks.__len__()
  songs= list()
  for i in range(num_tracks_i):
    segments=get_segments_by_track_id(spotify,album_tracks[i]['id'])
    songs.append(segments)
  return songs



