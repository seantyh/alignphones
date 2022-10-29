from typing import List, Optional
from dataclasses import dataclass
import numpy as np
from webvtt import Caption
from . import tree
from . import textgrids as tg
from . import align
from . import seq_algorithm as seqalgo
from . import utils
from .tree import AlignNode
from .utils import AlignContext, extract_dark_phones
from .align import EmitFrames

@dataclass
class AlignResult:    
    frames: EmitFrames
    align_epi_phones: AlignNode    
    align_char: AlignNode    
    align_utt: Optional[AlignNode] = None

    def to_textgrid(self, tg_path: str, minT, maxT):                
        tg.write_textgrid(
            tg_path, minT, maxT,
            self.frames,
            self.align_epi_phones,
            self.align_char,
            self.align_utt,            
            )

def align_vtt(
        vtt: List[Caption], 
        frames: EmitFrames, 
        context:AlignContext
    ) -> AlignResult:

    frames_new = []
    align_char = tree.create_root()
    align_epi_phones = tree.create_root()
    align_utt = tree.create_root()

    for vtt_x in vtt:    
        vtt_start = utils.to_seconds(vtt_x.start)
        vtt_end = utils.to_seconds(vtt_x.end)
        transcript = vtt_x.text
        frames_part = [x for x in frames
                if vtt_start < x.offset_s < vtt_end]
        align_result = align_transcript(
            transcript, frames_part, vtt_start, vtt_end, context)
        align_result.align_char.label = transcript
        align_char += align_result.align_char
        align_epi_phones += align_result.align_epi_phones
        frames_new.extend(frames_part)
        align_utt.children.append(align_char)

    return AlignResult(frames, 
                       align_epi_phones, 
                       align_char, 
                       align_utt)
        
def align_transcript(
        transcript: str, 
        frames: EmitFrames,
        vtt_start: float,
        vtt_end: float,
        context: AlignContext
    ):
            
    for frame_i in range(0, len(frames)-1):
        setattr(frames[frame_i], "start", frames[frame_i].offset_s)
        setattr(frames[frame_i], "end", frames[frame_i+1].offset_s)
        setattr(frames[0], "start", vtt_start)
        setattr(frames[-1], "end", vtt_end)
        if frames[frame_i].phone_token == "<blk>":
            phone_token = "[{}]".format(
                extract_dark_phones(frames[frame_i].phone_logits, context.allo_ipas))
        else:
            phone_token = frames[frame_i].phone_token
        setattr(frames[frame_i], "label", phone_token)

    # forced alignment
    # logit_mat: M (# of tokens) x V (# of phone set)
    logit_mat = np.vstack([x.phone_logits for x in frames])
    prob_mat = utils.softmax(logit_mat)    
    epi_ipas = context.epi.transliterate(transcript)
    epi_phones = context.ft.ipa_segs(epi_ipas)
    # M, V = prob_mat.shape
    # N = len(epi_phones)    
    _, backtrack = seqalgo.compute_trellis(prob_mat, epi_phones)

    # trace paths
    init_point = (logit_mat.shape[0]-1, len(epi_phones)-1)
    paths = align.create_path(init_point, backtrack)
    aligned_phones = align.align_epi_phones(paths, epi_phones, frames)
    aligned_chars = align.align_transcript(
                        transcript, 
                        aligned_phones.children, 
                        context.ft, 
                        context.epi)
    
    return AlignResult(frames, aligned_phones, aligned_chars)
    
