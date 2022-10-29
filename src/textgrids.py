from typing import List
from praatio import textgrid
from praatio.utilities.constants import Interval

from alignphones.src.align import EmitFrames

from .tree import AlignNode
from .utils import extract_dark_phones
from typing import Tuple

def make_textgrid_align_tree(
        nodes: List[AlignNode]
    ) -> List[Interval]:  
    
    entries = []

    for node_x in nodes:                
        if not (node_x.start and node_x.end): continue        
        entries.append((node_x.start, node_x.end, node_x.label))        
    return entries

def make_textgrid_emit_frames(
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
        char_tree: AlignNode, 
        epiphone_tree: AlignNode, 
        frames: EmitFrames, 
        allo_ipas: List[str], 
        minT: float, 
        maxT: float):

    tier_aligned_chars = make_textgrid_align_tree(char_tree.children)
    tier_aligned_phones = make_textgrid_align_tree(epiphone_tree.children)
    tier_raw_entries = make_textgrid_emit_frames(frames, allo_ipas)

    tg = textgrid.Textgrid()    
    tg.addTier(textgrid.IntervalTier("characs", entryList=tier_aligned_chars, minT=minT, maxT=maxT))
    tg.addTier(textgrid.IntervalTier("phones", entryList=tier_aligned_phones, minT=minT, maxT=maxT))
    tg.addTier(textgrid.IntervalTier("alloframe", entryList=tier_raw_entries, minT=minT, maxT=maxT))
    tg.save(tg_path, format="short_textgrid", includeBlankSpaces=False)