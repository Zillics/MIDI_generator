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

#Global hyperparameters
NOTE_RANGE = 36
SAMPLE_FREQUENCY = 1000
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
	for note in pretty_midi_obj.notes:
		note.pitch += k
	return pretty_midi_obj

# Determines which MIDI channel best represents melody, based on mean of pitch and entropy
def _extract_melody(instr_list):
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
	score = music21.converter.parse(midi_file)
	key = score.analyze('Krumhansl')
	key_num = pretty_midi.key_name_to_key_number(key.tonic.name.replace('-','b') + " Major")
	return key_num

# Extract the melody pattern from list of notes with the highest pitch
def _toMonophonic(notes):
	# Remove all notes that have ending after a higher note starts
	for note in notes:
		higher = [note_2 for note_2 in notes if ((note_2.start < note.end) & (note_2.pitch > note.pitch))]
		if(len(higher) > 0):
			notes.remove(note)
	return notes


def _midiToMatrix(pretty_midi_obj,sample_frequency,note_range):
	interval_s = 1/sample_frequency
	notes = _toMonophonic(pretty_midi_obj.notes)
	for note in notes:
		

def preprocess_pipeline(MIDI_src_folder):
	midi_src_directory = PATH_MIDI_SRC + MIDI_src_folder + '/**/*.mid'	
	midi_files = glob.glob(midi_src_directory, recursive=True)
	n = len(midi_files)
	for midi_file in midi_files:
		# 1. Determine key
		key = _determine_key(midi_file)
		# 2. Extract melody channel
		midi_obj = pretty_midi.PrettyMIDI(midi_file)
		melody_obj = _extract_melody(midi_obj.instruments)
		# 3. Transpose to C
		melody_obj = _transpose(melody_obj,-key)
		# 4. All notes to 0 - 35
		# 5. Convert to Numpy Matrix
		matrix = _midiToMatrix(melody_obj,SAMPLE_FREQUENCY,NOTE_RANGE)

if __name__ == '__main__':
	#Store user arguments in list
	arguments = sys.argv
	#Main function
	#transpose_to_common(arguments[1])
	preprocess_pipeline(arguments[1])