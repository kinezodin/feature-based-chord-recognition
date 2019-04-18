import keras
from keras import backend as K
import numpy as np
from keras.layers import LSTM, Dense, Input, TimeDistributed, Lambda, Add, Conv1D, Reshape, Average, Flatten
from keras.models import Model,load_model
from keras.optimizers import Adam
import pandas as pd
import os
from data_generators import data_generator as dg
from data_generators import precreate_dataset,predict_sequence,predict_sequence2
from keras.layers import Lambda
from keras import backend as K
from keras.callbacks import TensorBoard
from time import time
from sklearn.metrics import classification_report,accuracy_score

def create_MLP(num_feats):
	inp = Input((num_feats,))
	fc1 = Dense(60,activation='relu')(inp)
	fc2 = Dense(60,activation='relu')(fc1)
	fc3 = Dense(60,activation='relu')(fc2)
	o = Dense(25,activation='softmax')(fc3)
	m = Model(inp,o)
	m.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
	return m

def create_CNN(seqlen,num_feats):
	inp = Input((seqlen,num_feats))
	c = Conv1D(16,3,activation='relu',padding='same')(inp)
	c = Conv1D(16,3,activation='relu',padding='same')(c)
	c = Conv1D(16,3,activation='relu',padding='same')(c)
	c = Flatten()(c)
	c = Dense(25,activation='softmax')(c)
	m = model(inp,c)
	m.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
	return m



def create_LSTM(seq_len,num_feats):
	Reverse = Lambda(lambda x: K.reverse(x,axes=0))
	inp = Input((seq_len,num_feats))
	l1 = LSTM(128,return_sequences=True)(inp)
	o1 = l1
	l2 = LSTM(64,return_sequences=True)(o1)
	o2 = l2
	d = Dense(25,activation='softmax')
	o = TimeDistributed(d)(o2)
	m = Model(inp,o)
	m.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy',metrics=['acc'])
	return m

def create_bLSTM(seq_len,num_feats):
	Reverse = Lambda(lambda x: K.reverse(x,axes=0))
	inp = Input((seq_len,num_feats))
	rinp = Reverse(inp)
	l1 = LSTM(128,return_sequences=True)(inp)
	r1 = LSTM(128,return_sequences=True)(rinp)
	r1 = Reverse(r1)
	o1 = Add()([l1,r1])
	r2 = Reverse(o1)
	l2 = LSTM(64,return_sequences=True)(o1)
	r2 = LSTM(64,return_sequences=True)(r2)
	r2 = Reverse(r2)
	o2 = Add()([l2,r2])
	d = Dense(25,activation='softmax')
	o = TimeDistributed(d)(o2)
	m = Model(inp,o)
	m.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy',metrics=['acc'])
	return m

def create_ED(latent_dim,seq_len,num_feats):
	encoder_inputs = Input(shape=(seq_len, num_feats))
	encoder = LSTM(latent_dim, return_state=True,unroll=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)

	states = [state_h, state_c]

	decoder_inputs = Input(shape=(1, 25))
	decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,unroll=False)
	decoder_dense = Dense(25, activation='softmax')

	all_outputs = []
	inputs = decoder_inputs
	for _ in range(seq_len):
	    outputs, state_h, state_c = decoder_lstm(inputs,
	                                             initial_state=states)
	    outputs = decoder_dense(outputs)
	    all_outputs.append(outputs)

	    inputs = outputs
	    states = [state_h, state_c]

	decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['acc'])
	return model


# returns train, inference_encoder and inference_decoder models
def define_models_b(n_input, n_output, n_units,seq_len):
	# define training encoder
	encoder_inputs = Input(shape=(seq_len, n_input),name='encIn')
	encoderA = LSTM(n_units, return_state=False,return_sequences=True,name='enca')
	encoderB = LSTM(n_units, return_state=False,return_sequences=True,name='encb')
	encoderC = LSTM(n_units,return_state=True,name='encc')
	encoder_inputsA=encoderA(encoder_inputs)
	encoder_inputsB=encoderB(encoder_inputsA)
	encoder_outputs, state_h, state_c = encoderC(encoder_inputsB)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output),name='decIn')
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True,name='dec')
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(n_output, activation='softmax',name='decfc')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model

def define_models_c(n_input, n_output, n_units,seq_len):
	# define training encoder
	Reverse = Lambda(lambda x: K.reverse(x,axes=0))

	encoder_inputs = Input(shape=(seq_len, n_input),name='encIn')
	enc_in = Reverse()(encoder_inputs)
	encoderA = LSTM(n_units, return_state=False,return_sequences=True,name='enca')
	encoderB = LSTM(n_units, return_state=False,return_sequences=True,name='encb')
	encoderC = LSTM(n_units,return_state=True,name='encc')
	encoder_inputsA=encoderA(encoder_inputs)
	encoder_inputsB=encoderB(encoder_inputsA)
	encoder_outputs, state_h, state_c = encoderC(encoder_inputsB)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output),name='decIn')
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True,name='dec')
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(n_output, activation='softmax',name='decfc')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model


def create_EDcustom(latent_dim,seq_len,num_feats):
	Reverse = Lambda(lambda x: K.reverse(x,axes=0))

	encoder_inputs = Input(shape=(seq_len, num_feats))
	reverse_inp = Reverse(encoder_inputs)
	revencoder = LSTM(latent_dim,return_state=True,unroll=True)
	encoder = LSTM(latent_dim, return_state=True,unroll=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	rev_encoder_outputs, rev_state_h,rev_state_c = revencoder(reverse_inp)
	states = [Average()([state_h,rev_state_h]), Average()([state_c,rev_state_c])]

	encoder_outputs = Average()([encoder_outputs,rev_encoder_outputs])
	encoder_output = Dense(25,activation='softmax',name='First_chord_classifier')(encoder_outputs)

	#decoder_inputs = Input(shape=(1, 25))
	decoder_inputs = Reshape((1,25))(encoder_output)
	decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,unroll=False)
	decoder_dense = Dense(25, activation='softmax')

	all_outputs = []
	inputs = decoder_inputs
	for _ in range(seq_len):
	    outputs, state_h, state_c = decoder_lstm(inputs,
	                                             initial_state=states)
	    outputs = decoder_dense(outputs)
	    all_outputs.append(outputs)

	    inputs = outputs
	    states = [state_h, state_c]

	decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1),name='All_chord_classifier')(all_outputs)

	model = Model(encoder_inputs, [encoder_output,decoder_outputs])
	model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['acc'])
	return model


acc_dics = dict()
for feats in ['pitch',['key'],'all']:
	for mod in ['LSTM','bLSTM','CNN','ED_custom']:
		for seq_len in [10]:
			for augment in [True,False]:
			#seq_len=5
			
				batch_size=16
				acc_dics[str(mod)+str(seq_len)+str(augment)+str(feats)]=list()
				train_df = pd.read_csv('train_songsk.csv')
				test_df = pd.read_csv('test_songsk.csv')

				trainDG = dg(train_df, seq_len,batch_size,augment=augment, model=mod,features=feats)

				if mod=='ED_teacher':
					testDG = dg(test_df, seq_len,batch_size,augment=False, model='ED_noteacher',features=feats)
				else:
					testDG = dg(test_df, seq_len,batch_size,augment=False, model=mod,features=feats)
				
				tensorboard = TensorBoard(log_dir="logs/"+'aug'+str(augment)+mod+str(seq_len)+str(feats),batch_size=batch_size)

				if mod =='FC':
					x,y = next(trainDG)
					num_feats = x.shape[-1]
					m = create_MLP(num_feats)
				elif mod=='CNN':
					x,y = next(trainDG)
					num_feats = x.shape[-1]
					m = create_CNN(seq_len,num_feats)
				elif mod=='LSTM':
					x,y = next(trainDG)
					num_feats = x.shape[-1]
					m = create_LSTM(seq_len,num_feats)
				elif mod =='ED_custom':
					[x,x1],y=next(trainDG)
					num_feats = x.shape[-1]

					m, infenc, infdec = define_models_b(num_feats, 25, 128,seq_len)
					m.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['acc'])
				elif 'ED' in mod:
					x,_=next(trainDG)
					print(x.shape)
					num_feats = x.shape[-1]
					m = create_ED(64,seq_len,num_feats)
				elif mod=='bLSTM':
					x,y=next(trainDG)
					num_feats=x.shape[-1]
					m = create_bLSTM(seq_len,num_feats)
				
				m.summary()
				if mod!='ED_custom':
					his = m.fit_generator(trainDG,validation_data=testDG,steps_per_epoch=1000,epochs=20,validation_steps=100,callbacks=[tensorboard,])#,workers=2,use_multiprocessing=True)

				elif mod=='ED_custom':
					test_accs = list()
					for i in range(20):
						#print(str(his))
						print(i)
						his = m.fit_generator(trainDG,validation_data=testDG,steps_per_epoch=1000,epochs=1+i,validation_steps=10,callbacks=[tensorboard,],initial_epoch=i,workers=4,use_multiprocessing=True)
						cm=np.zeros((25,25))
						tdg = dg(test_df, seq_len,1,augment=False, model=mod,features=feats)
						y_true = list()
						y_pred = list()
						for j in range(1000):
							if j%100==0:
								print("progress: "+str(float(100.0*float(j)/float(1000))))
							[X1,X2],y=next(tdg)
							target=predict_sequence2(infenc, infdec, X1[0].reshape((1,X1.shape[1],X1.shape[2])), seq_len, 25)
							y_ind=np.argmax(y[0,:,:],1)
							t_ind=np.argmax(target[:,:],1)
							
							for k in range(y_ind.shape[0]):
								cm[y_ind[k],t_ind[k]]+=1
								y_true.append(y_ind[k])
								y_pred.append(t_ind[k])

						print(classification_report(y_true,y_pred))
						acc = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
						print(acc)
						test_accs.append(acc)
						acc_dics[str(mod)+str(seq_len)+str(augment)+str(feats)].append(acc)
						for ad in acc_dics:
							print(ad+" : "+str(acc_dics[ad]))

						print(test_accs)
						del tdg

				K.clear_session()

				del m
				del trainDG
				del testDG
				
		
