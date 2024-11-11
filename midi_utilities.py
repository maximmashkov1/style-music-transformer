import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

def get_midi_notes(path,track_idx=-1):
    midi_file = MidiFile(path)
    combined_notes = []
    try:
        tempo = next(msg.tempo for msg in midi_file.tracks[0] if msg.type == 'set_tempo')
    except:
        tempo = 500_000
    
    def process_track(track):
        elapsed_time = 0
        for msg in track:
            if msg.time > 0: 
                elapsed_time += (msg.time / midi_file.ticks_per_beat) * (tempo / 1_000_000)
            
            if hasattr(msg, 'channel') and msg.channel == 9:
                continue
            
            if msg.type == 'note_on' and msg.velocity > 0:
                if (msg.note, elapsed_time) not in combined_notes:
                    combined_notes.append((msg.note, elapsed_time))

    if track_idx==-1:
        for track in midi_file.tracks:
            process_track(track)
    else:
        process_track(midi_file.tracks[track_idx])
    
    combined_notes.sort(key=lambda x: x[1])
    
    return combined_notes





def create_midi_from_timings(note_timings, output_path):
    
    new_midi_file = MidiFile()
    for channel in note_timings:
        track = MidiTrack()
        new_midi_file.tracks.append(track)
        
        default_tempo = 500_000
        seconds_per_beat = default_tempo / 1_000_000 
        prev_off = {}
        prev_time_seconds = 0
        
        for i, (note, timing_seconds) in enumerate(channel):
            delta_time_seconds = timing_seconds - prev_time_seconds
            delta_time_ticks = int((delta_time_seconds / seconds_per_beat) * new_midi_file.ticks_per_beat)
            
            if len(prev_off.keys())!=0 and int(delta_time_ticks) > 0: 
                
                while True: #decay expiration time on expiring notes, off those that expire
                    #find smallest time
                    min_time=1e9
                    min_note=[]

                    for prev_note in prev_off.keys():
                        if prev_off[prev_note] < min_time:
                            min_time=prev_off[prev_note]
                            min_note=[prev_note]
                        elif prev_off[prev_note] == min_time:
                            min_note.append(prev_note)

                    if min_time < delta_time_ticks:
                        track.append(Message('note_off', note=min_note[0], time=min_time))
                        del prev_off[min_note[0]]
                        min_note.remove(min_note[0])
                        for note_ in min_note: #add all expiring at the same time
                            track.append(Message('note_off', note=note_, time=0))
                            del prev_off[note_]

                        for note_ in prev_off.keys(): #remove min time from everything else
                            prev_off[note_]-=min_time
                        delta_time_ticks-=min_time

                    else:
                        for note_ in prev_off.keys():
                            prev_off[note_]-=delta_time_ticks
                        break
            
            try: #end previous note if it's the same
                test=prev_off[note]
                track.append(Message('note_off', note=note, time=delta_time_ticks))
                delta_time_ticks=0
            except:
                pass
            track.append(Message('note_on', note=note, velocity=64, time=delta_time_ticks))
            prev_off[note] = 800
            prev_time_seconds = timing_seconds
    
    new_midi_file.save(output_path)

if __name__ == "__main__":

    input_midi_path = 'F:/datasets/lmd_full/b/b0af97d1877780518a34e24610dbb663.mid'
    tracks=[get_midi_notes(input_midi_path)]
    create_midi_from_timings(tracks, "real.mid")