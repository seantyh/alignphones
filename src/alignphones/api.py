from typing import List, Optional
from dataclasses import dataclass
from tqdm.auto import tqdm
import numpy as np
from webvtt import Caption
from . import tree
from . import textgrids as tg
from . import align
from . import seq_algorithm as seqalgo
from . import utils
from .tree import AlignNode, AlignNodeType
from .utils import AlignContext, extract_dark_phones, to_seconds
from .align import EmitFrames

@dataclass
class AlignResult:
    frames: EmitFrames
    align_epi_phones: AlignNode
    align_chars: AlignNode
    align_utts: Optional[AlignNode] = None

    def to_textgrid(self, tg_path: str, minT, maxT):
        minT = to_seconds(minT)
        maxT = to_seconds(maxT)
        
        tg.write_textgrid(
            tg_path, minT, maxT,
            self.frames,
            self.align_epi_phones,
            self.align_chars,
            self.align_utts,
            )

def align_vtt(
        vtt: List[Caption],
        frames: EmitFrames,
        context:AlignContext
    ) -> AlignResult:

    # `session` indicates the whole vtt (such as the whole wave file)
    session_frames = []    
    session_align_epi_phones = tree.create_node(AlignNodeType.Character)
    session_align_chars = tree.create_node(AlignNodeType.Utterance)
    session_align_utts = tree.create_node(AlignNodeType.Root)

    for vtt_x in tqdm(vtt):
        try:
            vtt_start = utils.to_seconds(vtt_x.start)
            vtt_end = utils.to_seconds(vtt_x.end)
            transcript = vtt_x.text
            frames_part = [x for x in frames
                    if vtt_start < x.offset_s < vtt_end]                
            align_result = align_transcript(
                transcript, frames_part, vtt_start, vtt_end, context)
            align_result.align_chars.label = transcript

            session_align_utts.children.append(align_result.align_chars)
            session_align_chars += align_result.align_chars
            session_align_epi_phones += align_result.align_epi_phones
            session_frames.extend(frames_part)
        except Exception as ex:
            print(ex)
        

    return AlignResult(session_frames,
                       session_align_epi_phones,
                       session_align_chars,
                       session_align_utts)

def align_transcript(
        transcript: str,
        frames: EmitFrames,
        vtt_start: float,
        vtt_end: float,
        context: AlignContext
    ):

    # set dynamic fields on frames
    if not isinstance(vtt_start, float):
        raise TypeError("vtt_start is not a float")
    
    if not isinstance(vtt_end, float):
        raise TypeError("vtt_end is not a float")

    for frame_i in range(0, len(frames)):
        setattr(frames[frame_i], "start", frames[frame_i].offset_s)
        if frame_i < len(frames)-1:
            setattr(frames[frame_i], "end", frames[frame_i+1].offset_s)
        else:
            setattr(frames[frame_i], "end", vtt_end)

        if frames[frame_i].phone_token == "<blk>":
            phone_token = "[{}]".format(
                extract_dark_phones(frames[frame_i].phone_logits, context.allo_ipas))
        else:
            phone_token = frames[frame_i].phone_token
        setattr(frames[frame_i], "label", phone_token)
    setattr(frames[0], "start", vtt_start)

    # forced alignment
    # logit_mat: M (# of tokens) x V (# of phone set)
    logit_mat = np.vstack([x.phone_logits for x in frames])
    prob_mat = utils.softmax(logit_mat)
    epi_ipas = context.epi.transliterate(transcript)
    epi_phones = context.ft.ipa_segs(epi_ipas)
    # M, V = prob_mat.shape
    # N = len(epi_phones)
    _, backtrack = seqalgo.compute_trellis(
                        prob_mat,
                        epi_phones,
                        context.ft,
                        context.allo_ipas)

    # trace paths
    init_point = (logit_mat.shape[0]-1, len(epi_phones)-1)
    paths = align.create_path(init_point, backtrack)
    aligned_phones = align.align_epi_phones(paths, epi_phones, frames)
    aligned_chars = align.align_characters(
                        transcript,
                        aligned_phones.children,
                        context.ft,
                        context.epi)

    return AlignResult(frames, aligned_phones, aligned_chars)

