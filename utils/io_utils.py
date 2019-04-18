import tensorflow as tf
import numpy as np
import csv
import glob


from utils import chord_utils,spotify_utils, matrix_utils

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def flatten_list(l):
	flat_list = [item for sublist in l for item in sublist]
	return flat_list

#Prints a dictionary indented
def print_dict(x,hp=False):
	for entry in x:
		if entry!="means" and entry!="stds":
			print('\t'+entry+" : "+str(x[entry]))


#Prints a list of dictionaries
def print_dict_list(x):
	for el in x:
		print_dict(el)

#Prints hyperparameters
def print_HParams(hparams):
	print_dict(hparams.values(),True)

#Stores a dictionary to a csv file
def store_dict_to_csv(in_dict,out_path):
	with open(out_path, 'w') as f:
		w = csv.DictWriter(f, in_dict.keys())
		w.writeheader()
		w.writerow(in_dict)

def format_numpy_arrays_in_dict(in_dict):
	ret_dict=dict()
	for entry in in_dict:
		if type(in_dict[entry]) is np.ndarray:
			ret_dict[entry]=in_dict[entry].tolist()
		else:
			ret_dict[entry]=in_dict[entry]
	return ret_dict



#Stores a list of dictionaries to a csv file
def store_dict_list_to_csv(in_list,out_path):
	with open(out_path,'w',newline='') as f:
		w=csv.DictWriter(f,in_list[-1].keys())
		w.writeheader()
		w.writerows(in_list)

#Loads a dictionary from a csv file
def load_dict_from_csv(csv_path):
	with open(csv_path,'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			return row

#Loads a list of dictionaries from a csv file
def load_dict_list_from_csv(csv_path):
	ret=list()
	with open(csv_path,'r') as f:
		reader=csv.DictReader(f)
		for row in reader:
			ret.append(row)
	return ret


#Loads an annotation file and returns as a list of dictionaries (keys: 'start','end','chord')
def load_annotations_file(filename,tab_separated=False,billboard=False):
    output_file = open(filename)
    output_data = output_file.read()
    split_output = output_data.split('\n')
    split_output = split_output[:-1]
    output_formated = list()
    for output_line in split_output:
        if tab_separated:
            start_time_str, end_time_str, chord_str = output_line.split('\t')
        elif billboard:
            #print('OUTPUTLINE IS:'+output_line)
            oll=output_line.split(' ')
            if oll[0]!='':
                output_line=oll[0]
                #print('NOW OUTPUTLINE IS:'+output_line)
                start_time_str, end_time_str, chord_str = output_line.split('\t')
        else:
            start_time_str, end_time_str, chord_str = output_line.split(' ')
        start_time = float(start_time_str)
        end_time = float(end_time_str)
        o_row = dict()
        o_row['start'] = start_time
        o_row['end'] = end_time
        o_row['chord'] = chord_str
        output_formated.append(o_row)
    return output_formated

def load_annotations_folder(folder_name,annotations_type):
	files=glob.glob(folder_name+"/*."+annotations_type)
	songs_chords=list()
	for file in files:
		if annotations_type=="chords":
			tab_separated=True
		else:
			tab_separated=False
		chords=load_annotations_file(file, tab_separated)
		songs_chords.append(chords)
	return songs_chords

#Combines a list of segments and a list of annotations, into a list of annotated segments (dictionary)
def combine_segments_annotations(segments,annotations):
	i = 0
	X = list()
	#print(song_segments.__len__())
	for j in range(annotations.__len__()-1):
		time = annotations[j]['start']
		while (time < annotations[j]['end'] and i < segments.__len__()):
			time = segments[i]['start']
			duration = float(segments[i]['duration'])
			if (time + duration < annotations[j]['end']):
				#row = duration, song_segments[i]['pitches'],song_segments[i]['timbre'], output_formated[j]['chord']
				row = segments[i]
				row['chord']=annotations[j]['chord']
			else:
				new_duration = annotations[j]['end'] - time
				if (2 * new_duration > duration):
					#row = duration, segments[i]['pitches'],segments[i]['timbre'], annotations[j]['chord']
					row = segments[i]
					row['chord']=annotations[j]['chord']
				else:
					#row = duration, segments[i]['pitches'],segments[i]['timbre'], annotations[j + 1]['chord']
					row=segments[i]
					row['chord']=annotations[j+1]['chord']
			X.append(row)

			i = i + 1
	return X


#Prints the feature vector x, the predicted value and the true value for 10 examples
def external_evaluation(x,y_pred,y_real):
	print("EXTERNAL EVALAUATION: here are 10 test set examples I got wrong:")
	correct_predictions=tf.equal(tf.argmax(y_pred,0),tf.argmax(y_real,0))
	#wrong_predictions=tf.equal(correct_predictions,0)
	correct_np=correct_predictions.eval()
	num_examples=x.shape[1]
	num_classes=y_pred.shape[0]
	print("CORRECT_NP IS:"+str(correct_np))
	randit=np.random.permutation(num_examples)
	num_evaluations=0
	for i in range(num_examples):
		if num_evaluations>10:
			break
		elif correct_np[randit[i]]!=1:
			print("#"+str(randit[i]))
			print("X:"+str(x[:,randit[i]]))
			if num_classes==13:
				chord_pred=chord_utils.class_to_string_tone(np.argmax(y_pred[:,randit[i]],0))
				chord_real=chord_utils.class_to_string_tone(np.argmax(y_real[:,randit[i]],0))
			else:
				chord_pred=chord_utils.class_to_string_tri(np.argmax(y_pred[:,randit[i]],0))
				chord_real=chord_utils.class_to_string_tri(np.argmax(y_real[:,randit[i]],0))
			num_evaluations+=1
			print("Real chord:"+chord_real)
			print("Predicted:"+chord_pred)


#annotations type= lab | chords
def download_albums(artist_name,annotations_path,annotations_type):
	store_path = "./data/songs/"+artist_name+"_"
	album_paths=glob.glob(annotations_path+"*")
	album_names=list()
	for i in range(album_paths.__len__()):
		if artist_name=="The Beatles":
			album_names.append(album_paths[i].split('-')[1])
			album_names[i]=album_names[i].replace("_"," ")[1:]
		elif artist_name=="Robbie Williams":
			album_names.append(album_paths[i].split("\\")[1])
		print("ALBUMPATH["+str(i)+"]:"+album_names[i])
	sp = spotify_utils.initialize_spotify()
	albums=spotify_utils.get_artist_albums(sp,artist_name)
	for i in range(albums.__len__()):
		print("Please enter the index of the album: '"+str(albums[i]['name'])+"' in the ALBUMPATH array above (or 'no' to skip album)")
		ind=input()
		if ind!='no':
			store_name=store_path+album_names[int(ind)]+'_'
			songs=spotify_utils.get_segments_for_album(sp,albums[i]['id'])
			songs_chords=load_annotations_folder(album_paths[int(ind)],annotations_type)
			if songs.__len__()!=songs_chords.__len__():
				print("NOT EQUAL NUMBER OF SONGS,abort")
				print(str(songs.__len__()))
				print(str(songs_chords.__len__()))
				return
			else:
				for j in range(songs.__len__()):
					print("durations:"+str(songs[j][-1]['start'])+', '+str(songs_chords[j][-1]['end']))
					#print("songs[j] is:"+str(songs[j]))
					new_store_name=store_name+str(j)+'.csv'
					combined=combine_segments_annotations(songs[j],songs_chords[j])
					store_dict_list_to_csv(combined,new_store_name)


def store_model(params,hparams,filename):
	merge_params_hparams={**params,**hparams.values()}
	merge_params_hparams = format_numpy_arrays_in_dict(merge_params_hparams)
	store_dict_to_csv(merge_params_hparams,filename)


#Load parameters and hyperparameters from stored .model file and return them as dictionaries
def load_model(filename):
	print("Loading stored model...")
	loadaded_dict=load_dict_from_csv(filename)
	params=dict()
	hparams=dict()
	if loadaded_dict["model"]=="feedforward":
		num_layers=int(loadaded_dict["num_layers"])+1
		for i in range(1,num_layers+1):
			params['W'+str(i)]= matrix_utils.string_to_numpy_array(loadaded_dict['W'+str(i)])
			params['b'+str(i)]=matrix_utils.string_to_numpy_array(loadaded_dict['b'+str(i)])
			print("PARAMS["+'W'+str(i)+" shape is:"+str(params['W'+str(i)].shape))
		
	elif loadaded_dict["model"]=="convolutional":
		params["W1"] = np.array(loadaded_dict["W1"])
		params["W2"] = np.array(loadaded_dict["W2"])
		params["Wfc"]= np.array(loadaded_dict["Wfc"])
		params["bfc"]= np.array(loadaded_dict["bfc"])

	for key in loadaded_dict:
			if key not in params:
				hparams[key]=loadaded_dict[key]


	return params,hparams