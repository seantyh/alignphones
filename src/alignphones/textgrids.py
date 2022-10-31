from typing import List, Optional
from praatio import textgrid
from praatio.utilities.constants import Interval

from .align import EmitFrames

from .tree import AlignNode
from .utils import extract_dark_phones

def make_textgrid_align_tree(
        nodes: List[AlignNode]
    ) -> List[Interval]:  
    
    entries = []

    for node_x in nodes:                
        if not (node_x.start and node_x.end): continue        
        entries.append((node_x.start, node_x.end, node_x.label))        
    return entries

def make_textgrid_emit_frames(
        frames: EmitFrames
    ):
    entries = []
    for i in range(len(frames)):
        start = frames[i].offset_s        
        end = frames[i].end   # type: ignore
        label = frames[i].label # type: ignore
        entries.append((start, end, label))
    return entries

def write_textgrid(
        tg_path: str, 
        minT: float, 
        maxT: float,
        frames: EmitFrames, 
        epiphone_tree: AlignNode, 
        char_tree: AlignNode,         
        utt_tree: Optional[AlignNode] = None,
    ):

    tg = textgrid.Textgrid()
    tier_raw_entries = make_textgrid_emit_frames(frames)
    tg.addTier(textgrid.IntervalTier("alloframe", entryList=tier_raw_entries, minT=minT, maxT=maxT))
    if epiphone_tree:
        tier_aligned_phones = make_textgrid_align_tree(epiphone_tree.children)
        tg.addTier(textgrid.IntervalTier("phones", entryList=tier_aligned_phones, minT=minT, maxT=maxT))
    
    if char_tree:
        tier_aligned_chars = make_textgrid_align_tree(char_tree.children)
        tg.addTier(textgrid.IntervalTier("characs", entryList=tier_aligned_chars, minT=minT, maxT=maxT))        

    if utt_tree:
        tier_aligned_utt = make_textgrid_align_tree(utt_tree.children)
        tg.addTier(textgrid.IntervalTier("utterances", entryList=tier_aligned_utt, minT=minT, maxT=maxT))                            
    
    tg.save(tg_path, format="short_textgrid", includeBlankSpaces=False)