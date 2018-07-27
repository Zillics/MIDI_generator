import pretty_midi
import music21
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

'''
def transpose_to_common(MIDI_src_folder,dest_dir):
	midi_src_path = PATH_MIDI_SRC + MIDI_src_folder + '/*.mid'
	midi_files = glob.glob(midi_src_path)
	keys = []
	for midi_file in midi_files:
		print(midi_file)
		score = music21.converter.parse(midi_file)
		key = score.analyze('Krumhansl')
		key_num = pretty_midi.key_name_to_key_number(key.tonic.name.replace('-','b') + " Major")
		print(key,key_num) 
		keys.append((midi_file,key_num))
		transposed_path = PATH_MIDI_SRC + MIDI_src_folder + '/transposed/'
	os.makedirs(os.path.dirname(transposed_path),exist_ok=True)
	for (midi_file,key_num) in keys:
		dest_path = dest_dir + os.path.basename(midi_file)
		if(key_num != 0):
			print('Dest path: %s' % (dest_path))
			_transpose(midi_file,-key_num,dest_path)
		else:
			copyfile(midi_file,dest_path)
'''

def _transpose(pretty_midi_obj,k):
	print("Transposing.")
	for note in pretty_midi_obj.notes:
		note.pitch += k
	return pretty_midi_obj

# Determines which MIDI channel best represents melody, based on mean of pitch and entropy
def _extract_melody(instr_list):
	print("Extracting melody channel.")
	n = len(instr_list)
	means = np.zeros(n)
	norm_entr = np.zeros(n)
	w = 2.5 # Weight for how much to take entropy into account vs mean pitch
	i = 0
	for instrument in instr_list:
		# Pitch average
		note_matrix = np.array(instr.notes) # Convert one hot encoding to single integers
		means[i] = note_matrix.mean()
		# Normalized entropy
		hist = matrix[:,0:128].sum(axis=0)
		p = hist/hist.sum()
		norm_entr[i] = entropy(p)/(np.log2(p.shape[0]))
		i += 1
	# Heuristics function of melody (means: 0 - 127, norm_entr: 0 - 1)
	heur = means + 127*w*norm_entr
	melody_idx = np.argmax(heur)
	return instr_list[melody_idx]

def _determine_key(midi_file_path):
	print("Determining key: ")
	score = music21.converter.parse(midi_file)
	key = score.analyze('Krumhansl')
	print(key)
	key_num = pretty_midi.key_name_to_key_number(key.tonic.name.replace('-','b') + " Major")
	return key_num

# Extract the melody pattern from list of notes (might be polyphonic) with the highest pitch
def _toMonophonic(notes):
	print("Converting to monophonic.")
	# Remove all notes that have ending after a higher note starts
	for note in notes:
		higher = [note_2 for note_2 in notes if ((note_2.start < note.end) & (note_2.pitch > note.pitch))]
		if(len(higher) > 0):
			notes.remove(note)
	return notes

# Convert one MIDI channel into compressed numpy matrix with monophonic one hot encoded notes + duration + rests
def _midiToMatrix(pretty_midi_obj,sample_frequency,n_octaves):
	print("Converting MIDI to matrix.")
	interval_s = 1/sample_frequency
	notes = _toMonophonic(pretty_midi_obj.notes)
	n_notes = len(notes)
	note_range = 12*n_octaves
	n_features = note_range + 2
	matrix = np.zeros((n_notes,n_features))
	i = 0
	prev_note = notes[0]
	for note in notes:
		# p = pitch, d = duration, r = rest before note
		p = note.pitch % note_range
		d = note.end - note.start
		r = note.start - prev_note.end
		matrix[i,p] = 1
		matrix[i,note_range] = d
		matrix[i,note_range + 1] = r
		prev_note = note
		i += 1
	return matrix

def _matrixToMidi(matrix,dest_path):
	print("Converting matrix to MIDI")
	# Create a PrettyMIDI object
	midi_obj = pretty_midi.PrettyMIDI()
	# Create an Instrument instance for a Piano instrument
	piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
	piano = pretty_midi.Instrument(program=piano_program)
	t = 0.0
	note_range = matrix.shape[1] - 2
	notes = [ np.where(r==1)[0][0] for r in matrix[:,0:note_range] ] # Convert one hot encoding to single integers
	n_notes = len(notes) # Number of consecutive notes
	print("Writing %d notes to MIDI file" % (n_notes))
	for i in range(0,n_notes):
		p = notes[i]
		rest = matrix[i,note_range + 1] # Rest before note
		s = t + rest # Start = current time + rest before note
		e = s + matrix[i,note_range] #End = start + duration of note in seconds
		note = pretty_midi.Note(velocity=100, pitch=p, start=s, end=e)
		piano.notes.append(note)
		t = e
	midi_obj.instruments.append(piano)
	midi_obj.write(dest_path)

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
	data_idx = len(glob.glob(PATH_DATA + MIDI_src_folder + '/'))
	output_directory = PATH_DATA + MIDI_src_folder + data_idx + '/'
	os.makedirs(os.path.dirname(output_directory),exist_ok=True)
	os.makedirs(os.path.dirname(output_directory + 'MIDI/'),exist_ok=True)
	os.makedirs(os.path.dirname(output_directory + 'matrix/'),exist_ok=True)
	midi_files = glob.glob(midi_src_directory, recursive=True)
	n = len(midi_files)
	file_idx = 0
	for midi_file in midi_files:
		# 1. Determine key
		key = _determine_key(midi_file)
		# 2. Extract melody channel
		midi_obj = pretty_midi.PrettyMIDI(midi_file)
		melody_obj = _extract_melody(midi_obj.instruments)
		# 3. Transpose to C
		melody_obj = _transpose(melody_obj,-key)
		# 4. All notes to 0 - 35 and convert to numpy Matrix
		matrix = _midiToMatrix(melody_obj,SAMPLE_FREQUENCY,N_OCTAVES)
		# 5. Generate MIDI sample of matrix
		midi_sample_path = output_directory + 'MIDI/' + str(file_idx) + os.path.basename(midi_file)
		_matrixToMidi(matrix,midi_sample_path)
		# 6. Store matrix in numpy file
		numpy_filepath = output_directory + 'matrix/' + str(file_idx) + os.path.basename(midi_file)[:-4]
		_matrixToNumpy(matrix,numpy_filepath)

		file_idx += 1
	# 7. Generate one tensor of all generated matrices in sequence form suitable for LSTM network
	matrix_directory = PATH_DATA + 'matrix/'
	_generate_sequence_tensor(matrix_directory,PATH_TRAINING_DATA)

if __name__ == '__main__':
	#Store user arguments in list
	arguments = sys.argv
	#Main function
	#transpose_to_common(arguments[1])
	preprocess_pipeline(arguments[1])