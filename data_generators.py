import pandas as pd
import numpy as np


def string_to_class_tri(old_chord_str):
    ret=0


    chord_str=old_chord_str.replace("\n",'')
    s=chord_str.replace('\'','')
    chord_str=s

    s = chord_str.replace(' ','')
    chord_str=s
    chord_str=chord_str.replace('\t','')
    chord_str=chord_str.replace(':','')
    #print('New chord str is:'+chord_str)

    if chord_str[0] == 'A':
        if chord_str.__len__()>1 and chord_str[1]=='m' and chord_str[2]=='i' and chord_str[3]=='n':
            ret=1
        elif chord_str.__len__()>1 and chord_str[1]=='#':
            if chord_str.__len__()>2 and  chord_str[2]=='m' and chord_str[3]=='i' and chord_str[4]=='n':
                ret=3
            else:
                ret=4
        elif chord_str.__len__()>1 and chord_str[1]=='b':
            if chord_str.__len__()>2 and chord_str[2] == 'm' and chord_str[3] == 'i' and chord_str[4] == 'n':
                ret=23
            else:
                ret=24
        else:
            ret=2

    elif chord_str[0]=='B':
        if chord_str.__len__()>1 and chord_str[1]=='m' and chord_str[2]=='i' and chord_str[3]=='n':
            ret=5
        elif chord_str.__len__()>1 and chord_str[1]=='b':
            if chord_str.__len__()>2 and chord_str[2]=='m' and chord_str[3]=='i' and chord_str[4]=='n':
                ret=3
            else:
                ret=4
        else:
            ret=6

    elif chord_str[0]=='C':
        if chord_str.__len__()>1 and chord_str[1] == 'm' and chord_str[2] == 'i' and chord_str[3] == 'n':
            ret = 7
        elif chord_str.__len__()>1 and chord_str[1] == '#':
            if chord_str.__len__()>2 and chord_str[2] == 'm' and chord_str[3] == 'i' and chord_str[4] == 'n':
                ret = 9
            else:
                ret = 10
        else:
            ret = 8

    elif chord_str[0]=='D':
        if chord_str.__len__()>1 and chord_str[1]=='m' and chord_str[2]=='i' and chord_str[3]=='n':
            ret=11
        elif chord_str.__len__()>1 and chord_str[1]=='#':
            if chord_str.__len__()>2 and chord_str[2]=='m' and chord_str[3]=='i' and chord_str[4]=='n':
                ret=13
            else:
                ret=14
        elif chord_str.__len__()>1 and chord_str[1]=='b':
            if chord_str.__len__()>2 and chord_str[2] == 'm' and chord_str[3] == 'i' and chord_str[4] == 'n':
                ret=9
            else:
                ret=10
        else:
            ret=12

    elif chord_str[0]=='E':
        if chord_str.__len__()>1 and chord_str[1]=='m' and chord_str[2]=='i' and chord_str[3]=='n':
            ret=15
        elif chord_str.__len__()>1 and chord_str[1]=='b':
            if chord_str.__len__()>2 and chord_str[2]=='m' and chord_str[3]=='i' and chord_str[4]=='n':
                ret=13
            else:
                ret=14
        else:
            ret=16

    elif chord_str[0]=='F':
        if chord_str.__len__()>1 and chord_str[1]=='m' and chord_str[2]=='i' and chord_str[3]=='n':
            ret=17
        elif chord_str.__len__()>1 and chord_str[1]=='#':
            if chord_str.__len__()>2 and chord_str[2]=='m' and chord_str[3]=='i' and chord_str[4]=='n':
                ret=19
            else:
                ret=20
        else:
            ret=18

    elif chord_str[0]=='G':
        if chord_str.__len__()>1 and chord_str[1] == 'm' and chord_str[2] == 'i' and chord_str[3] == 'n':
            ret = 21
        elif chord_str.__len__()>1 and chord_str[1] == '#':
            if chord_str.__len__()>2 and chord_str[2] == 'm' and chord_str[3] == 'i' and chord_str[4] == 'n':
                ret = 23
            else:
                ret = 24
        elif chord_str.__len__()>1 and chord_str[1] == 'b':
            if chord_str.__len__()>2 and chord_str[2] == 'm' and chord_str[3] == 'i' and chord_str[4] == 'n':
                ret = 19
            else:
                ret = 20
        else:
            ret = 22

    #print("ret="+str(ret))
    return ret

def shift_1(xj):
	x_ret = np.zeros(xj.shape)
	
	for i in range(1,12):
		x_ret[:,i-1]=xj[:,i]
	x_ret[:,11] = xj[:,0]
	
	return x_ret

def shift_by(pitches,n):
	x_ret = pitches
	for i in range(n):
		x_ret=shift_1(x_ret)
	return x_ret

def y_one_semitone_down(Y_np):
    indexes=np.argmax(Y_np,axis=1)
    i_length=indexes.shape[0]
    shifted_indexes=np.zeros((i_length,1))
    y_hot = np.zeros((i_length,25))
    
    for i in range(i_length):
        if indexes[i]!=0:
            if indexes[i]==1:
                shifted_indexes[i]=23
            elif indexes[i]==2:
                shifted_indexes[i]=24
            else:
                shifted_indexes[i]=indexes[i]-2
            y_hot[i,int(shifted_indexes[i])]=1
        else:
            y_hot[i,0]=1
    return y_hot

def shift_y_by(y,n):
	yy = y
	for i in range(n):
		yy = y_one_semitone_down(yy)
	return yy


def precreate_dataset(df, seq_len, model,augment,size=0,features=['duration','confidence','timbre','key','key_confidence','mode','mode_confidence','loudness_start','loudness_max_time','loudness_max','new_bar','new_beat','new_sect']):
	if size==0:
		size = int(df.shape[0]/seq_len)
	datagen=data_generator(df, seq_len,size,augment=augment, model=model,features=features)
	x,y = next(datagen)
	return x,y


def data_generator(df, seq_len,batch_size, model='FC',augment=False,features=['duration','confidence','timbre','key','key_confidence','mode','mode_confidence','loudness_start','loudness_max_time','loudness_max','new_bar','new_beat','new_sect']):
	if 'all' in features:
		features=['duration','confidence','timbre','key','key_confidence','mode','mode_confidence','loudness_start','loudness_max_time','loudness_max','new_bar','new_beat','new_sect']
	elif 'pitch' in features:
		features=[]

	num_features = 12

	for f in features:
		if f=='timbre' or f=='key':
			num_features+=12
		else:
			num_features+=1

	means = pd.read_csv('means.csv')
	stds = pd.read_csv('stds.csv')
	while(True):
		x = np.zeros((batch_size,seq_len,num_features))
		y = np.zeros((batch_size,seq_len,25))
		rp = np.random.permutation(df.shape[0]-seq_len)
		#print('permute')
		for i in range(batch_size):
			#print(str(float(i/batch_size)))
			for j in range(seq_len):
				#x[i,j] = [float(p) for p in df['pitches'][rp[i]+j][1:-1].split(',')]
				chrd = string_to_class_tri(df['chord'][rp[i]+j])
				for p in range(12):
					x[i,j,p] = (df['pitch_'+str(p)][rp[i]+j]-0.5)/0.5
				curr_idx = 12
				if 'key' in features:
					for p in range(12):
						x[i,j,curr_idx+p] = df['key'+str(p)][rp[i]+j]
					curr_idx+=12
				if 'timbre' in features:
					for p in range(12):
						x[i,j,curr_idx+p] = (df['timbre_'+str(p)][rp[i]+j]-means['timbre_'+str(p)][0])/stds['timbre_'+str(p)]
					curr_idx+=12
				if 'mode' in features:
					x[i,j,curr_idx] = df['mode'][rp[i]+j]
					curr_idx+=1
				for f in features:
					if f not in ['pitches','timbre','key','mode']:
						x[i,j,curr_idx] = (df[f][rp[i]+j]-means[f][0])/stds[f][0]
						curr_idx+=1
				y[i,j,chrd]=1

			if augment:
				shift = np.random.permutation(12)[0]
				y[i] = shift_y_by(y[i],shift)
				x[i,:,:12] = shift_by(x[i,:,:12],shift)
				if 'key' in features:
					x[i,:,12:24] = shift_by(x[i,:,12:24],shift)
		
		if model=='ED_teacher':
			decoder_input_data = np.zeros((batch_size, 1, 25))
			decoder_input_data[:, 0, :] = y[:,0,:]

			yield [x,decoder_input_data],y
		elif model=='ED_noteacher':
			decoder_input_data = np.zeros((batch_size, 1, 25))
			yield [x,decoder_input_data],y
		elif model=='ED_custom':
			yield get_data_ED2(x,y,num_features)
		elif model=='FC':
			yield x.reshape((x.shape[0],-1)),y[:,int(seq_len/2),:].reshape((y.shape[0],25))
		elif model=='CNN':
			yield x, y[:,int(seq_len/2),:].reshape((y.shape[0],25))
		else:
			yield x,y


def get_data_ED(X_train,Y_train,num_features):
	X1=np.zeros((X_train.shape[0],X_train.shape[1],num_features+25))
	X1[:,:,:num_features]=X_train
	zer=np.zeros((Y_train.shape[0],Y_train.shape[2]))
	Y_sc=Y_train[:,:-1,:]
	X2=np.zeros(X1.shape)
	X2[:,0,num_features:]=zer
	X2[:,1:,num_features:]=Y_sc
	y=np.zeros(X1.shape)
	y[:,:,num_features:]=Y_train
	return [X1,X2],y

def get_data_ED2(X_train,Y_train,num_features):
	X1=np.zeros((X_train.shape[0],X_train.shape[1],num_features+25))
	X1[:,:,:num_features]=X_train
	zer=np.zeros((Y_train.shape[0],Y_train.shape[2]))
	Y_sc=Y_train[:,:-1,:]
	X2=np.zeros(X1.shape)
	X2[:,0,num_features:]=zer
	X2[:,1:,num_features:]=Y_sc
	X1 = X1[:,:,:num_features]
	X2 = X2[:,:,num_features:]
	return [X1,X2],Y_train


# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input
	target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
		yhat[0,0,:(cardinality-25)] = source[0,t,:(cardinality-25)]
		output.append(yhat[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = yhat
	return np.array(output)

def predict_sequence2(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input
	target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = yhat
	return np.array(output)