from itertools import groupby
from typing import Tuple, List
from epitran import Epitran
from panphon import FeatureTable
from allosaurus.emit_frame import EmitFrameInfo
from .seq_algorithm import BacktrackTable, TrellisLoc, AlignPath
from .tree import AlignNode, AlignNodeType, create_root
from . import tree as tree

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

def align_transcript(
        transcript: str,
        epi_nodes: List[AlignNode],
        ft: FeatureTable,
        epi: Epitran
    ) -> AlignNode:
    
    cursor = 0
    root = tree.create_root()    
    for ch_idx, ch_ipas in enumerate(epi.transliterate_char(transcript)): # type: ignore
        char_phones_x = ft.ipa_segs(ch_ipas)
        first_phone_i = cursor
        last_phone_i = cursor+len(char_phones_x)+1        
        char_node = AlignNode(
            AlignNodeType.Character,
            transcript[ch_idx],
            epi_nodes[first_phone_i: last_phone_i]
            )
        root.children.append(char_node)        
    return root
    
def align_epi_phones(
        paths: AlignPath,
        epi_phones: List[str],
        frames: EmitFrames
    ) -> AlignNode:

    label_spans = groupby(paths, key=lambda x: x[1])
    aligned = []
    last_frame_i = -1
    root = create_root()

    for epi_id, paths_x in label_spans:
        epi_phone_x = epi_phones[epi_id]
        paths_x = list(paths_x)
        # print(label_id, paths_x)
        start_frame_i = paths_x[0][0]
        end_frame_i = paths_x[-1][0]

        if last_frame_i == start_frame_i:
            # skip duplicate frames, i.e. dropping labels
            empty_node = AlignNode(
                AlignNodeType.EpiPhone,
                epi_phone_x,
                []
            )
            root.children.append(empty_node)
            continue
        else:
            frame_nodes = [
                AlignNode(
                    AlignNodeType.AlloPhone,
                    frame_x.phone_token,
                    [],
                    frame_x.start, frame_x.end)  # type: ignore
                for frame_x 
                in frames[start_frame_i:end_frame_i+1]
            ]

            phone_node = AlignNode(
                AlignNodeType.EpiPhone,
                epi_phone_x,
                frame_nodes
            )
            root.children.append(phone_node)

        last_frame_i = start_frame_i
    return root