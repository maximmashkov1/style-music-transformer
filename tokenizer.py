import torch

class MusicalTokenizer:
    def __init__(self, notes=128, time_shifts=125, longest_time_token=500):
         
        self.time_shifts = time_shifts
        self.notes = notes
        self.longest_time_token = longest_time_token
        self.div = self.longest_time_token // time_shifts 
        
        note_on_vocab = [f"note_on_{i}" for i in range(self.notes)]
        time_shift_vocab = [f"time_shift_{i}" for i in range(self.time_shifts)]
        self.vocab = ['<pad>'] + note_on_vocab + time_shift_vocab + ['<start>', '<end>']
    
    def tokenize(self, sequence, add_delimiters=True, add_initial_time=False, pad=0):
        """
        Tokenizes a sequence of [abs_note, abs_time], with time in seconds.
        Returns a torch long tensor
        """
        output = [] if not add_initial_time else self.__time_to_events(self.round_(max(0, sequence[0][1]) * 1000))
        seq_len = len(sequence)

        for idx, (abs_note, abs_time) in enumerate(sequence):
            output.append(self.vocab[abs_note + 1])
            
            if abs_time != 0 and idx < seq_len - 1:
                next_time = sequence[idx + 1][1]
                delta_time = self.round_(max(0, next_time - abs_time) * 1000) 
                output.extend(self.__time_to_events(delta_time))
        
        if add_delimiters:
            output = ['<start>'] + output + ['<end>']
        
        output.extend(['<pad>'] * pad)
        
        output = torch.tensor([self.vocab.index(t) for t in output], dtype=torch.long)
        return output
    
    def detokenize(self, sequence):
        sequence=sequence.flatten().cpu().detach().tolist()
        sequence=[self.vocab[i] for i in sequence]
        output = []
        time_to_add = 0
        while sequence:
            token = sequence.pop(0)
            
            if token.startswith('note'):
                output.append([self.vocab.index(token) - 1])
            
            elif token.startswith('time'):
                times = [token]
                while sequence and sequence[0].startswith('time'):
                    times.append(sequence.pop(0))
                
                time = self.__events_to_time(times) / 1000
                if len(output)==0:
                    time_to_add+=time
                else:
                    output[-1].append(time)
                
            
            elif token.startswith('<'):
                continue
        
        abs_output = []
        cum_time = 0
        for item in output:
            rel_time = item[1] if len(item) > 1 else 0
            abs_output.append([item[0], cum_time+time_to_add])
            cum_time += rel_time
        
        return abs_output

    def get_empty_sequence(self, length):
        return torch.tensor([self.vocab.index('<start>')] + [self.vocab.index('<pad>')]*(length-1), dtype=torch.long)

    
    def __time_to_events(self, delta_time):
        """
        Translate accumulated delta_time between midi events into vocab using time_cutter
        event_list and index_list are passed by reference, so nothing is returned.
        Pass-by-reference is necessary to execute this function within a loop.
    
        Args:
            delta_time (int): time between midi events
            event_list (list): accumulated vocab event list during midi translation
            index_list (list): accumulated vocab index list during midi translation
            _vocab (list, optional): vocabulary list to translate into
        """
    
        time = self.__time_cutter(delta_time)
        time_events=[]
        for i in time:
            idx = self.notes + i
            time_events.append(self.vocab[idx])
        return time_events
    
    def __events_to_time(self, events):
        return (sum([self.vocab.index(event)-self.notes for event in events]))* self.div
        
    def __time_cutter(self, time):
        """
        As per Oore et. al, 2018, the time between midi events must be expressed as a sequence of finite-length
        time segments, so as to avoid considering every possible length of time in the vocab. This sequence can be
        expressed as k instances of a maximum time shift followed by a leftover time shift, i.e.,
        time = k * max_time_shift + leftover_time_shift
        where k = time // max_time_shift; leftover_time_shift = time % max_time_shift
    
        This function will translate the input time into indices in the vocabulary then cut it as above
    
        Args:
            time (int > 0): input milliseconds to translate and cut
            lth (int, optional): max milliseconds to consider for vocab, i.e., max_time_shift
            div (int, optional): number of ms per time_shift;
                       lth // div = num_time_shift_events
    
        Returns:
            time_shifts (list): list of time shifts into which time is cut
                                each time_shift is in range: (1, lth // div); 0 is not considered
        """
        time_shifts = []
    
        # assume time = k * lth, k >= 0; add k max_time_shifts (lth // div) to time_shifts
        for i in range(time // self.longest_time_token):
            time_shifts.append(self.round_(self.longest_time_token / self.div))   # custom round for consistent rounding of 0.5, see below
        leftover_time_shift = self.round_((time % self.longest_time_token) / self.div)
        time_shifts.append(leftover_time_shift) if leftover_time_shift > 0 else None
    
        return time_shifts
    
    def compute_tokenization_error(self, x):
        """
        Used to compare different parameters
        """
        processed = self.detokenize(self.tokenize(x, add_initial_time=True))
        return sum([abs(t[1]-p[1]) for (t,p) in zip(x,processed)])/len(x)
        
    
    def round_(self, a):

        b = a // 1
        decimal_digits = a % 1
        adder = 1 if decimal_digits >= 0.5 else 0
        return int(b + adder)


