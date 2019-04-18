from utils import spotify_utils,io_utils


all_paths=list()

cp='./data/annotations/Carole King Annotations/Tapestry/'
all_paths.append(['54ashUPqW5v7L533WeZ4cW',cp+'01 I Feel The Earth Move.lab'])
all_paths.append(['32e0HRB05GSibDgv7fkYwh',cp+'02 So Far Away.lab'])
all_paths.append(['0XF4zGa9UNzwQN05F9GUKN',cp+"03 It's Too Late.lab"])
all_paths.append(['3yylO3r9fnCt2tLSlPQcOb',cp+"04 Home Again.lab"])
all_paths.append(['58KeNIwSX4njezDBQhRFRJ',cp+"05 Beautiful.lab"])
all_paths.append(['7jDbTugWAx8h3vDheUjs2b',cp+"06 Way Over Yonder.lab"])
all_paths.append(['3veFg0px7FnV6J3g0GYbc0',cp+"07 You've Got A Friend.lab"])

def download():
	sp = spotify_utils.initialize_spotify()
	store_path = "./data/songs/king_"
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
