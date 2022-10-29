from dataclasses import dataclass
from typing import List, Optional
from enum import Enum, auto

from .align import AlignedEntry

class AlignNodeType(Enum):
    Root = auto()
    Character = auto()
    EpiPhone = auto()
    AlloPhone = auto()

@dataclass
class AlignNode:
    nodetype: AlignNodeType
    label: str    
    children: List["AlignNode"]
    frame_start: Optional[float] = None
    frame_end: Optional[float] = None

    @property
    def start(self):
        if not self.frame_start:       
            chd = self.children
            return (chd and chd[0].start) or None
        else:
            return self.frame_start
    
    @property
    def end(self):
        if not self.frame_end:
            chd = self.children
            return (chd and chd[-1].end) or None
        else:
            return self.frame_end        
        

def create_root() -> AlignNode:
    root = AlignNode(
        AlignNodeType.Root, 
        "root",
        [])
    return root
