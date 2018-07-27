import pretty_midi
from music21 import converter, instrument, note, chord, stream
import numpy as np
from scipy.stats import entropy

import pickle
import glob
import re
import os
from shutil import copyfile
import sys

PATH_MIDI_SRC = 'src_midi/'
PATH_DATA = 'data/'
PATH_TRAINING_DATA = PATH_DATA + 'training_data/'

#Global hyperparameters
N_OCTAVES = 3
SAMPLE_FREQUENCY = 1000
SEQUENCE_LENGTH = 25

def _transpose(note_list,k):
	print("Transposing")
	lowest = 127
	note_list_output = []
	for note_i in note_list:
		if(note_i.pitch.midi < lowest):
			note_l = note_i
			lowest = note_i.pitch.midi
	lowest_octave = note_l.pitch.implicitOctave # Find octave of the lowest note in list
	for note_i in note_list:
		note_i.pitch = note_i.pitch.transpose(k-(lowest_octave+1)*12)
	return 0

# Determines which MIDI channel best represents melody, based on mean of pitch and entropy
def _extract_melody(midi_obj):
	print("Extracting melody channel")
	n = len(midi_obj) # Number of parts
	means = np.zeros(n)
	norm_entr = np.zeros(n)
	w = 1.5 # Weight for how much to take entropy into account vs mean pitch
	i = 0
	for part in midi_obj:
		# Pitch average
		elements = list(part.flat.notes)#[note.pitch for note in instrument.notes]
		pitch_list = [element.pitch.midi for element in elements if isinstance(element,note.Note)]
		means[i] = np.array(pitch_list).mean()
		# Normalized entropy
		hist = np.zeros(128)
		for idx in range(0,128):
			hist[idx] = pitch_list.count(idx)
		p = hist/hist.sum()
		norm_entr[i] = entropy(p)/(np.log2(p.shape[0]))
		i += 1

	# Heuristics function of melody (means: 0 - 127, norm_entr: 0 - 1)
	heur = means + 127*w*norm_entr
	melody_idx = np.argmax(heur)
	print(melody_idx)
	note_list = [element for element in list(midi_obj[melody_idx].flat.notes) if isinstance(element,note.Note)]
	return note_list

def _determine_key(midi_obj):
	print("Determining key: ")
	key = midi_obj.analyze('Krumhansl')
	print(key)
	key_num = pretty_midi.key_name_to_key_number(key.tonic.name.replace('-','b') + " Major")
	return key_num

# Extract the melody pattern from list of notes (might be polyphonic) with the highest pitch
def _toMonophonic(note_list):
	print("Converting to monophonic")
	# Remove all notes that have ending after a higher note starts
	for note_1 in note_list:
		higher = [note_2 for note_2 in note_list if ((note_2.offset == note_1.offset) & (note_2.pitch.midi > note_1.pitch.midi))]
		if(len(higher) > 0):
			note_list.remove(note_1)
	return note_list

# Convert one MIDI channel into compressed numpy matrix with monophonic one hot encoded notes + duration + rests
def _midiToMatrix(note_list,sample_frequency,n_octaves):
	print("Converting MIDI to matrix")
	interval_s = 1/sample_frequency
	note_list = _toMonophonic(note_list)
	n_notes = len(note_list)
	note_range = 12*n_octaves
	n_features = note_range + 2
	matrix = np.zeros((n_notes,n_features))
	i = 0
	for i in range(0,n_notes):
		last = i == n_notes-1
		# p = pitch, d = duration, r = rest before note
		p = note_list[i].pitch.midi % note_range
		print("%d mod %d = %d" % (note_list[i].pitch.midi,note_range,p))
		end_i = note_list[i].offset + note_list[i].duration.quarterLength
		start_i = note_list[i].offset 
		if(not last):
			if(end_i > note_list[i+1].offset):
				d = note_list[i+1].offset - start_i
				r = 0
			else:
				d = end_i - start_i
				r = note_list[i+1].offset - end_i
		else:
			d = end_i - start_i
		matrix[i,p] = 1
		matrix[i,note_range] = d
		if(not last):
			matrix[i+1,note_range + 1] = r
		i += 1
	return matrix

def _matrixToMidi(matrix,dest_path,transpose_octaves):
	print("Converting matrix to MIDI")
	# Create a PrettyMIDI object
	#midi_obj = pretty_midi.PrettyMIDI()
	# Create an Instrument instance for a Piano instrument
	piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
	piano = pretty_midi.Instrument(program=piano_program)
	t = 0.0
	note_range = matrix.shape[1] - 2
	note_list = [ np.where(r==1)[0][0] for r in matrix[:,0:note_range] ] # Convert one hot encoding to single integers
	n_notes = len(note_list) # Number of consecutive notes
	output_notes = []
	for i in range(0,n_notes):
		p = note_list[i] + transpose_octaves*12
		rest = matrix[i,note_range + 1] # Rest before note
		s = t + rest # Start = current time + rest before note
		d = matrix[i,note_range]
		t = s + d #End = start + duration of note in seconds
		new_note = note.Note()
		new_note.pitch.midi = p
		#print(p)
		new_note.quarterLength = d
		new_note.offset = s
		new_note.storedInstrument = instrument.Piano()
		output_notes.append(new_note)
	midi_stream = stream.Stream(output_notes)
	midi_stream.write('midi', fp=dest_path)
	print("%d notes written to MIDI file" % (n_notes))

def _matrixToNumpy(matrix,dest_path):
	with open(dest_path, 'wb') as filepath:
		pickle.dump(matrix, filepath)

def _generate_sequence_tensor(matrix_directory,tensor_directory):
	files = glob.glob(matrix_directory + '*')
	n_features = N_OCTAVES*12 + 2
	n_sequences = 0
	X_all = np.zeros((n_sequences,SEQUENCE_LENGTH,n_features),dtype=np.float32)
	y_all = np.zeros((n_sequences,n_features),dtype=np.float32)
	k = 0
	for file in files:
		with open(file, 'rb') as filepath:
			matrix = np.array(pickle.load(filepath),dtype=np.float32)
		try:
			(X,y) = _prepare_sequences(matrix,SEQUENCE_LENGTH)
		except Exception as e:
			print(e)
			print("Preparing sequences failed. Skipping this track.")
			k += 1
		else:
			n_sequences += X.shape[0]
			X_all = np.append(X_all,X,axis=0)
			y_all = np.append(y_all,y,axis=0)
			k += 1
			print("Sequences so far: %d : " % (n_sequences))
			print("Completed: %d / %d" % (k,n))


def _prepare_sequences(matrix,sequence_length):
	if(matrix.shape[0] < sequence_length + 1):
		raise ValueError("Matrix contains less notes than sequence length (%d vs %d)" % (matrix.shape[0],sequence_length))
	n_sequences = matrix.shape[0] - sequence_length
	n_features = matrix.shape[1]

	X = np.zeros((n_sequences,sequence_length,n_features),dtype=np.float32)
	y = np.zeros((n_sequences,n_features),dtype=np.float32)

	for i in range(0, n_sequences):
		sequence_in = matrix[i:i + sequence_length,:]
		sequence_out = matrix[i + sequence_length,:]
		X[i,:,:] = sequence_in
		y[i,:] = sequence_out
	return (X,y)


def preprocess_pipeline(MIDI_src_folder):
	midi_src_directory = PATH_MIDI_SRC + MIDI_src_folder + '/**/*.mid'
	data_idx = len(glob.glob(PATH_DATA + MIDI_src_folder + '*/'))
	output_directory = PATH_DATA + MIDI_src_folder + str(data_idx) + '/'
	os.makedirs(os.path.dirname(output_directory),exist_ok=True)
	os.makedirs(os.path.dirname(output_directory + 'MIDI/'),exist_ok=True)
	os.makedirs(os.path.dirname(output_directory + 'matrix/'),exist_ok=True)
	midi_files = glob.glob(midi_src_directory, recursive=True)
	n = len(midi_files)
	file_idx = 0
	print("Processing MIDI files in %s. Dumping results in %s....." % (midi_src_directory,output_directory))
	for midi_file in midi_files:
		print("MIDI file %s, %d / %d" % (midi_file,file_idx,n))
		midi_obj = converter.parse(midi_file)
		# 1. Determine key
		key = _determine_key(midi_obj)
		# 2. Extract melody channel
		melody_notes = _extract_melody(midi_obj)
		# 3. Transpose to C
		_transpose(melody_notes,-key)
		#for note_i in melody_notes:
		#	print((note_i.pitch,note_i.pitch.implicitOctave))
		# 4. All notes to 0 - 35 and convert to numpy Matrix
		matrix = _midiToMatrix(melody_notes,SAMPLE_FREQUENCY,N_OCTAVES)
		# 5. Generate MIDI sample of matrix
		midi_sample_path = output_directory + 'MIDI/' + str(file_idx) + os.path.basename(midi_file)
		_matrixToMidi(matrix,midi_sample_path, 4)
		# 6. Store matrix in numpy file
		numpy_filepath = output_directory + 'matrix/' + str(file_idx) + os.path.basename(midi_file)[:-4]
		_matrixToNumpy(matrix,numpy_filepath)

		file_idx += 1
	# 7. Generate one tensor of all generated matrices in sequence form suitable for LSTM network
	#matrix_directory = PATH_DATA + 'matrix/'
	#_generate_sequence_tensor(matrix_directory,PATH_TRAINING_DATA)

if __name__ == '__main__':
	#Store user arguments in list
	arguments = sys.argv
	#Main function
	#transpose_to_common(arguments[1])
	preprocess_pipeline(arguments[1])