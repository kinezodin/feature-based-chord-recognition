from utils import io_utils,spotify_utils

all_paths=list()

cp='./data/annotations/Queen Annotations/chordlab/Queen/'

all_paths.append(['2OBofMJx94NryV2SK8p8Zf',cp+'Greatest Hits I/01 Bohemian Rhapsody.lab'])
all_paths.append(['4xPnIniRbg40s1jCD246pc',cp+'Greatest Hits I/02 Another One Bites The Dust.lab'])
all_paths.append(['0CQddjWDflPd0dOqqVkJ2H',cp+'Greatest Hits I/04 Fat Bottomed Girls.lab'])
all_paths.append(['5FYslb39kAXnPmxSsJyzC5',cp+'Greatest Hits I/05 Bicycle Race.lab'])
all_paths.append(['5NpLjebPN2sTg5TYT3jEqq',cp+"Greatest Hits I/06 You're My Best Friend.lab"])
all_paths.append(['0LPMWPwCHrIorXSdpnPVyv',cp+"Greatest Hits I/07 Don't Stop Me Now.lab"])
all_paths.append(['54BcazwvRS8abIqivDKipj',cp+'Greatest Hits I/08 Save Me.lab'])
all_paths.append(['3DuBPMUKNtdrFUt5Yu9Yqg',cp+'Greatest Hits I/09 Crazy Little Thing Called Love.lab'])
all_paths.append(['251aFD82kJ2eCfvozNQB6b',cp+'Greatest Hits I/10 Somebody To Love.lab'])
all_paths.append(['4PhVskrP2ZSGsWqYaaw7nu',cp+'Greatest Hits I/12 Good Old-Fashioned Lover Boy.lab'])
all_paths.append(['6Ir4U5SFeIgUUmei82CiBg',cp+'Greatest Hits I/13 Play The Game.lab'])
all_paths.append(['5u99eyhU43QoLeXXG6caoL',cp+'Greatest Hits I/15 Seven Seas Of Rhye.lab'])
all_paths.append(['3OYsVAtg2j28YkfchSUfwv',cp+'Greatest Hits I/16 We Will Rock You.lab'])
all_paths.append(['2mbk9uIlbFUEeYof7I361V',cp+'Greatest Hits I/17 We Are The Champions.lab'])
all_paths.append(['6G5hkdh0Tijc4NhjUMNCCe',cp+'Greatest Hits II/01 A Kind Of Magic.lab'])
all_paths.append(['59CsbfwLNCAUGbBQV3Tki4',cp+'Greatest Hits II/04 I Want It All.lab'])
all_paths.append(['4ue5ET9msGNJSO6sSbrCVE',cp+'Greatest Hits II/05 I Want To Break Free.lab'])
all_paths.append(['2u8KG85xyQTLuVP0m0Nqw7',cp+'Greatest Hits II/09 Who Wants To Live Forever.lab'])
all_paths.append(['6so5hTwXfzsCTzmKoA04UV',cp+'Greatest Hits II/14 Hammer To Fall.lab'])
all_paths.append(['5RjjRF8MEnoe8aD2hcbQl6',cp+'Greatest Hits II/15 Friends Will Be Friends.lab'])

def download():
	sp = spotify_utils.initialize_spotify()
	store_path = "./data/songs/queen_"
	#io_utils.print_dict_list(billboard_index)
	i=0
	for p in all_paths:
			print("i="+str(i))
			#print("p:"+str(p))
			track_id= p[0]
			track_annotations_path= p[1]
			track_info = track_annotations_path.split('/')
			track_title = track_info[-1][3:-4]
			store_name = store_path+'-'+track_title+'.csv'
			segments = spotify_utils.get_segments_by_track_id(sp,track_id)
			chords = io_utils.load_annotations_file(track_annotations_path,tab_separated=False,billboard=True)
			combined=io_utils.combine_segments_annotations(segments,chords)
			io_utils.store_dict_list_to_csv(combined,store_name)
