import logging
import os
import re
from collections import deque
from enum import Enum, unique
from html.parser import HTMLParser
from typing import List, Tuple, Optional

from lxml import html


# Logger
LOGGER = logging.getLogger()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))


# font-size mapping
FONT_SIZE_MAPPING = {
    # Official definition
    'xx-small': 1,
    'x-small': 2,
    'small': 3,
    'medium': 4,
    'large': 5,
    'x-large': 6,
    'xx-large': 7,

    # Customized definition
    'smaller': 3,
    'larger': 5,
}


@unique
class TagNodeType(Enum):
    START = 'START'
    END = 'END'
    START_END = 'START_END'
    TEXT = 'TEXT'


def is_valid_tag(tag):
    if ':' in tag or tag in ['br']:
        return False
    return True


class TagNode:

    def __init__(self, id_, name, type_, start_offset, pair_node_id: int = -1, depth: int = -1, style: dict = None):
        self.id_ = id_
        self.name = name
        self.type_ = type_
        self.start_offset = start_offset
        self.pair_node_id = pair_node_id

        # Depth of a node, only START and START_END nodes will have this value.
        self.depth = depth

        # Style of a node
        self.style = style or {}

        # Normalize style on initialization
        self.set_style(self.normalize_style())

    def set_pair_node_id(self, pair_node_id: int):
        self.pair_node_id = pair_node_id

    def set_depth(self, depth: int):
        self.depth = depth

    def get_style(self, normalize=False):
        """Get style.

        :param normalize: This may be deprecated now, as normalization is executed on initialization.
        :return: A dict.
        """
        # TODO: Justify whether param normalize can be deprecated.
        if normalize:
            return self.normalize_style()

        return self.style

    def set_style(self, style: dict):
        self.style = style

    def normalize_style(self):
        """Some rules will be applied here to generate final styles."""
        # Unify style keys
        style_key_mapping = {
            'align': 'alignment',
            'text-align': 'alignment',
            'page-break-before': 'page-break',
            'page-break-after': 'page-break',
        }

        # Map some tags to corresponding styles
        placeholder_style_value = '1'  # For some cases, we only care about the style key.
        tag_to_style_mapping = {
            'b': {
                'font-weight': 'bold',
            },
            'i': {
                'italic': placeholder_style_value,
            },
            'img': {
                'img': placeholder_style_value,
            },
            'page': {
                'page-break': placeholder_style_value,
            },
            'center': {
                'alignment': 'center',
            },
        }

        norm_style = {}
        for key, value in self.style.items():
            norm_style[style_key_mapping.get(key, key)] = value

        if self.name in tag_to_style_mapping.keys():
            norm_style.update(tag_to_style_mapping.get(self.name))

        # Compute font-size for HTML4 <font> tag.
        if self.name == 'font' and 'size' in self.style.keys():
            norm_style['font-size'] = norm_style['size']
            norm_style.pop('size')

        return norm_style

    def merge_style(self, other_style: dict):
        for key, value in other_style.items():
            # The following style don't need to be merged
            if key == 'display' or key.startswith('xmlns'):
                continue
            if key not in self.style:
                self.style[key] = value

        self._compute_relative_font_size_style(other_style)

    def update_type(self, new_type: str):
        self.type_ = new_type

    def get_start_offset(self):
        return self.start_offset

    def get_end_offset(self, doc: str, bound: int):
        """Compute the exclusive end_offset of a closed / self-closed DOM node tag."""
        if self.type_ == TagNodeType.START.value:
            raise Exception(f'Invalid operation for tag type: {self.type_}')

        if self.type_ == TagNodeType.END.value:
            tag_len = len(self.name) + 3
        elif self.type_ == TagNodeType.START_END.value:
            if bound == -1:
                sub_doc = doc[self.start_offset:]
            else:
                sub_doc = doc[self.start_offset: bound]
            tag_len = sub_doc.find('>') + 1
        else:
            sub_doc = doc[self.start_offset: bound].rstrip()
            tag_len = len(sub_doc)

        return self.start_offset + tag_len

    def _compute_relative_font_size_style(self, other_style: dict):
        if self.style.get('font-size', '').endswith('%') and other_style.get('font-size'):
            percentage = float(self.style['font-size'].rstrip('%'))
            parent_font_size = other_style.get('font-size')
            parent_font_size_value = re.search(r'[0-9.]+', parent_font_size).group()
            if not parent_font_size_value:
                return

            new_font_size_value = round(float(parent_font_size_value) * percentage / 100, 2)
            new_font_size_unit = parent_font_size[len(parent_font_size_value):]
            new_font_size = f'{new_font_size_value}{new_font_size_unit}'
            self.style['font-size'] = new_font_size

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'TagNode[id_=%s, name=%s, type_=%s, start_offset=%s, pair_node_id=%s, depth=%s]' % (
            self.id_, self.name, self.type_, self.start_offset, self.pair_node_id, self.depth
        )


class Style:

    def __init__(self, raw_style: List[Tuple]):
        self.raw_style = raw_style

    def to_dict(self):
        dict_ = {}

        if self.raw_style is None:
            return dict_

        try:
            for key, value in self.raw_style:
                if key != 'style':
                    dict_[key] = value
                    continue

                style_entries = value.split(';')
                for entry in style_entries:
                    cleaned_entry = entry.strip()
                    if cleaned_entry:
                        items = cleaned_entry.split(':')
                        if len(items) != 2:
                            continue
                        entry_key, entry_value = items
                        dict_[entry_key.strip()] = (entry_value or '').strip()
        except Exception as e:
            LOGGER.error('raw_style: %s', self.raw_style)
            LOGGER.error(e, exc_info=True)

        return dict_


class Block:

    def __init__(self, start_offset: int, end_offset: int, html_str: str, start_node_id: int = -1, end_node_id: int = -1):
        self.start_offset = start_offset  # Inclusive offset
        self.end_offset = end_offset  # Exclusive offset
        self.html = html_str  # Raw HTML content

        # Style info
        self.attrs = {}

        self.label = 'none'

        self.start_node_id = start_node_id

        self.end_node_id = end_node_id

    def set_attrs(self, **kwargs):
        self.attrs.update(kwargs)

    def get_label(self) -> str:
        return self.label

    def set_label(self, label):
        self.label = label

    def get_html(self):
        """Get raw HTML that current node represents in the DOM tree."""
        return self.html

    def get_start_node_id(self):
        return self.start_node_id

    def get_end_node_id(self):
        return self.end_node_id

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Block[%s:%s](%s:%s)' % (self.start_offset, self.end_offset, self.start_node_id, self.end_node_id)


class SANHtmlParser(HTMLParser):
    TAG_NODE_START_ID = 0

    def __init__(self):
        super().__init__()

        # Init data structures
        self.nodes_list = []
        self.nodes_stack = deque()

        # ID generator
        self.ids_pool = self._id_generator()

        # HTML content
        self.content = None

        # Line start offsets
        self.line_start_offsets = []

        # Parent node' depth
        self.parent_node_depth = -1

    def feed(self, data):
        self.content = data
        self.line_start_offsets = self._get_line_start_offsets_v2()

        super().feed(data)

        self._adjust_depth()

        self._compute_style()

        if not self.get_nodes_list():
            raise Exception('Cannot parse input document.')

    def handle_starttag(self, tag, attrs):
        if not is_valid_tag(tag):
            return
        tag_node = self._build_tag_node(tag, TagNodeType.START.value, self.getpos(), attrs=attrs)

        # Depth
        tag_node.set_depth(self.parent_node_depth + 1)
        self.parent_node_depth += 1

        self.nodes_stack.append(tag_node)

    def handle_endtag(self, tag):
        if not is_valid_tag(tag):
            return
        tag_node = self._build_tag_node(tag, TagNodeType.END.value, self.getpos())

        while self.nodes_stack:
            last_tag_node = self.nodes_stack.pop()
            if last_tag_node.name == tag and last_tag_node.type_ == TagNodeType.START.value:  # A matching start tag
                tag_node.set_pair_node_id(last_tag_node.id_)
                last_tag_node.set_pair_node_id(tag_node.id_)
                tag_node.set_depth(last_tag_node.depth)

                # Update parent_node_depth
                self.parent_node_depth -= 1

                break

    def handle_startendtag(self, tag, attrs):
        if not is_valid_tag(tag):
            return
        tag_node = self._build_tag_node(tag, TagNodeType.START_END.value, self.getpos(), attrs=attrs)

        # Set depth
        tag_node.set_depth(self.parent_node_depth + 1)
        # Set pair node ID
        tag_node.set_pair_node_id(tag_node.id_)

    def handle_data(self, data):
        # Skip if String is empty
        if re.fullmatch('\s*', data):
            return

        tag_node = self._build_tag_node(data.strip(), TagNodeType.TEXT.value, self.getpos())
        tag_node.set_depth(self.parent_node_depth)
        tag_node.set_pair_node_id(tag_node.id_)

    def error(self, message):
        LOGGER.error('SANHtmlParser Error: %s', message)

    @classmethod
    def _id_generator(cls):
        next_id = cls.TAG_NODE_START_ID
        while True:
            yield next_id
            next_id += 1

    def _build_tag_node(self, name, type_, pos: Tuple[int, int], attrs: List[Tuple] = None):
        tag_id = next(self.ids_pool)
        start_offset = self._get_global_offset(pos)
        tag_node = TagNode(tag_id, name, type_, start_offset, style=Style(attrs).to_dict())

        self.nodes_list.append(tag_node)

        return tag_node

    def _get_line_start_offsets_v2(self) -> List[int]:
        sep = '\n'
        lines = self.content.split(sep)
        cur_offset = 0
        line_start_offsets = [0] * len(lines)
        line_sep_len = len(sep)
        for line_no in range(len(lines) - 1):
            line_len = len(lines[line_no]) + line_sep_len
            line_start_offsets[line_no + 1] = cur_offset + line_len
            cur_offset += line_len

        return line_start_offsets

    def _get_global_offset(self, pos: Tuple[int, int]) -> int:
        line_no = pos[0] - 1

        return self.line_start_offsets[line_no] + pos[1]

    def get_nodes_list(self) -> List[TagNode]:
        return self.nodes_list

    def _adjust_depth(self):
        """Depth should be adjusted due to the existence of unbalanced tags."""
        acc_diff = 0  # Accumulated depth difference from real value.
        for node in self.get_nodes_list():
            if node.type_ != TagNodeType.END.value:
                node.set_depth(node.depth - acc_diff)

            if node.type_ == TagNodeType.START.value and node.pair_node_id == -1:
                # Convert an unclosed tag to a self-closed tag
                node.update_type(TagNodeType.START_END.value)
                acc_diff += 1

    def _compute_style(self):
        """Compute style data for all nodes.

        A child node's style will inherit from parent nodes' styles.
        """
        try:
            nodes_stack = deque()
            for node in self.get_nodes_list():
                if node.type_ == TagNodeType.END.value:
                    # Validate
                    if (not nodes_stack) or (
                            nodes_stack[-1].name != node.name and nodes_stack[-1].type_ != TagNodeType.START.value):
                        raise Exception('Tag mismatch!')

                    # Pop stack
                    nodes_stack.pop()
                else:
                    # Compute style
                    if nodes_stack:
                        last_tag_node = nodes_stack[-1]
                        node.merge_style(last_tag_node.get_style(normalize=True))

                    if node.type_ == TagNodeType.START.value:
                        # Push to stack
                        nodes_stack.append(node)
        except Exception as e:
            LOGGER.warning('Computing styles information for tags failed.')
            LOGGER.error(e, exc_info=True)

    def build_block(self, start_node: TagNode):
        end_node_id = start_node.pair_node_id

        # Return None if there is no text in the block
        if TagNodeType.TEXT.value not in [node.type_ for node in self.nodes_list[start_node.id_: end_node_id + 1]]:
            return None, end_node_id

        exclusive_end_offset = self.nodes_list[end_node_id].get_end_offset(self.content, self.nodes_list[end_node_id + 1].start_offset)
        block = Block(
            start_node.start_offset,
            exclusive_end_offset,
            self.content[start_node.start_offset: exclusive_end_offset],
            start_node_id=start_node.id_,
            end_node_id=end_node_id,
        )
        block.set_attrs(**start_node.get_style(normalize=True))
        return block, end_node_id

    def surround_with_pair_tags(self, current_start_node: TagNode):
        current_end_node = self.nodes_list[current_start_node.pair_node_id]
        if self.nodes_list[current_start_node.id_ - 1].pair_node_id == current_end_node.id_ + 1:
            return True
        return False

    def to_blocks(self) -> List[Block]:
        R2L_BLOCKING_TAGS = ['table', 'title']
        L2R_BLOCKING_TAGS = ['div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']

        body_start, body_end = self.get_body_node_ids()
        blocks = []
        try:
            node_idx = 0
            last_idx = 0
            while node_idx < len(self.nodes_list):

                # Skip the nodes that are not in the html body
                if node_idx < body_start or node_idx > body_end:
                    node_idx += 1
                    continue

                node = self.nodes_list[node_idx]

                if node.type_ == TagNodeType.START.value and node.name in R2L_BLOCKING_TAGS:
                    # Build a block
                    block, end_node_id = self.build_block(node)
                    # Only append valid block
                    if block:
                        blocks.append(block)
                    node_idx, last_idx = end_node_id + 1, end_node_id + 1

                elif node.type_ == TagNodeType.TEXT.value:
                    start_node_id = -1
                    # Handle the case text surrounded without any tags
                    if self.nodes_list[node.id_-1].type_ == TagNodeType.END.value:
                        start_node_id = node.id_

                    # Search for an appropriate start node
                    else:
                        for pointer in range(node_idx - 1, last_idx - 1, -1):
                            if any([
                                self.nodes_list[pointer].name not in L2R_BLOCKING_TAGS,
                                self.nodes_list[pointer].style.get('display') == 'inline',
                                self.surround_with_pair_tags(self.nodes_list[pointer]),
                            ]):
                                continue
                            start_node_id = pointer
                            break

                    if start_node_id == -1:
                        raise RuntimeError(f'Error, Cannot find any appropriate block! {node_idx}')

                    block, end_node_id = self.build_block(self.nodes_list[start_node_id])
                    blocks.append(block)
                    node_idx, last_idx = end_node_id + 1, end_node_id + 1

                # elif node.type_ == TagNodeType.START_END.value:
                #     block, end_node_id = self.build_block(node)
                #     blocks.append(block)
                #     node_idx, last_idx = end_node_id + 1, end_node_id + 1
                else:
                    node_idx += 1

        except Exception as e:
            LOGGER.error(e, exc_info=True)

        return blocks

    def get_body_node_ids(self):
        for idx, node in enumerate(self.nodes_list):
            if node.type_ == TagNodeType.START.value and node.name == 'body':
                return node.id_, node.pair_node_id + 1
        raise RuntimeError('Can not find the body tag in the html!')

    def get_content(self):
        return self.content
