from itertools import groupby
from .seq_algorithm import BacktrackTable, TrellisLoc, AlignPath
from epitran import Epitran
from panphon import FeatureTable
from typing import Tuple, DefaultDict, List
from allosaurus.emit_frame import EmitFrameInfo

EmitFrames = List[EmitFrameInfo]
EntryLabel = str  # IpaPhones or character
AlignedEntry = Tuple[str, EmitFrames]
AlignedEntries = List[AlignedEntry]

def create_path(init_point: TrellisLoc, backtrack: BacktrackTable):
    path = []    
    buf = [init_point]
    while buf:
        cur_point = buf.pop()
        path.append(cur_point)
        if cur_point in backtrack:
            prev_point = backtrack[cur_point][0]            
            buf.append(prev_point)            
    return path[::-1]            

def align_phones(
        paths: AlignPath, 
        epi_phones: List[str], 
        frames: EmitFrames
        ) -> AlignedEntries:
    label_spans = groupby(paths, key=lambda x: x[1])
    aligned = []
    last_frame_i = -1
    for epi_id, paths_x in label_spans:
        epi_phone_x = epi_phones[epi_id]
        paths_x = list(paths_x)
        # print(label_id, paths_x)
        start_frame_i = paths_x[0][0]
        end_frame_i = paths_x[-1][0]

        if last_frame_i == start_frame_i:
            # skip duplicate frames, i.e. dropping labels
            aligned.append((epi_phone_x, [])) 
            continue
        else:
            aligned.append((epi_phone_x, 
                           frames[start_frame_i:end_frame_i+1]))        
        last_frame_i = start_frame_i
    return aligned

def align_characters(
        transcript: str, 
        aligned_phones: AlignedEntries, 
        ft: FeatureTable, 
        epi: Epitran
        ) -> AlignedEntries:        
    char_map = []
    cursor = 0
    for ch_idx, ch_ipas in enumerate(epi.transliterate_char(transcript)): # type: ignore
        char_phones_x = ft.ipa_segs(ch_ipas)        
        char_map.append(list(range(cursor, cursor+len(char_phones_x)+1)))
        cursor += len(char_phones_x)        
    
    aligned = []
    for char_i, phone_idxs in enumerate(char_map):             
        if not phone_idxs:
            continue
        first_phone_i = phone_idxs[0]
        last_phone_i = phone_idxs[-1]
        frames_x = [aligned_phones[i][1]
                    for i in range(first_phone_i, last_phone_i)]
        frames_x = sum(frames_x, [])        
        aligned.append((transcript[char_i], frames_x))
    return aligned