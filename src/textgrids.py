from typing import List
from praatio import textgrid
from .paths import AlignedEntries, EmitFrames
from .utils import extract_dark_phones

def make_aligned_textgrid(aligned_entries: AlignedEntries):        
    entries = []
    for label_x, frames_x in aligned_entries:                
        if not frames_x: continue
        start = frames_x[0].offset_s    
        end = frames_x[-1].end  # type: ignore
        entries.append((start, end, label_x))        
    return entries

def make_raw_textgrid(
        frames: EmitFrames, 
        allo_ipas: List[str]
    ):
    entries = []
    for i in range(len(frames)):
        start = frames[i].offset_s        
        end = frames[i].end   # type: ignore
        if frames[i].phone_token == "<blk>":
            phone_token = "[{}]".format(extract_dark_phones(frames[i].phone_logits, allo_ipas))
        else:
            phone_token = frames[i].phone_token
        entries.append((start, end, phone_token))
    return entries

def write_textgrid(
        tg_path: str, 
        aligned_chars: AlignedEntries, 
        aligned_phones: AlignedEntries, 
        frames: EmitFrames, 
        allo_ipas: List[str], 
        minT: float, 
        maxT: float):

    tier_aligned_chars = make_aligned_textgrid(aligned_chars)
    tier_aligned_phones = make_aligned_textgrid(aligned_phones)
    tier_raw_entries = make_raw_textgrid(frames, allo_ipas)

    tg = textgrid.Textgrid()    
    tg.addTier(textgrid.IntervalTier("characs", entryList=tier_aligned_chars, minT=minT, maxT=maxT))
    tg.addTier(textgrid.IntervalTier("phones", entryList=tier_aligned_phones, minT=minT, maxT=maxT))
    tg.addTier(textgrid.IntervalTier("alloframe", entryList=tier_raw_entries, minT=minT, maxT=maxT))
    tg.save(tg_path, format="short_textgrid", includeBlankSpaces=False)