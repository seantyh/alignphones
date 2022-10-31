from dataclasses import dataclass
from typing import List, Optional, Union
from enum import Enum, auto

class AlignNodeType(Enum):
    Root = auto()
    Utterance = auto()
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

    def __repr__(self):        
        return ("<AlignNode[{type_name}] {start}-{end}: {label} ({nchild} children)>"
                .format(type_name=self.nodetype.name,
                        start=self.start,
                        end=self.end,
                        label=self.label,
                        nchild=len(self.children)))

    def __iadd__(self, other: "AlignNode"):
        if not isinstance(other, AlignNode):
            raise TypeError("AlignNode expected")
        if self == other:
            raise ValueError("Add to itself")
        self.children.extend(other.children)
        return self

    def __getitem__(self, idx):
        return self.children[idx]

    def __iter__(self):
        return iter(self.children)

    @property
    def start(self) -> Union[float, None]:
        if not self.frame_start:       
            starts = [x.start
                      for x in self.children
                      if x.start]
            return starts[0] if starts else None
        else:
            return self.frame_start
    
    @property
    def end(self) -> Union[float, None]:
        if not self.frame_end:
            ends = [x.end
                    for x in self.children
                    if x.end]
            return ends[-1] if ends else None
        else:
            return self.frame_end        
        

def create_node(node_type=AlignNodeType.Root) -> AlignNode:
    node = AlignNode(
        node_type, 
        node_type.name,
        [])
    return node
