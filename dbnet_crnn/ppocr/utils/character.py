import numpy as np
import string


class CharacterOps(object):
    """ Convert between text-label and text-index """

    def __init__(self, config):
        self.character_type = config['character_type']
        self.loss_type = config['loss_type']
        self.max_text_len = config['max_text_length']
        if self.character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif self.character_type == "ch":
            character_dict_path = config['character_dict_path']
            add_space = False
            if 'use_space_char' in config:
                add_space = config['use_space_char']
            self.character_str = ""
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if add_space:
                self.character_str += " "
            dict_character = list(self.character_str)
        elif self.character_type == "en_sensitive":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        else:
            self.character_str = None
        assert self.character_str is not None, \
            "Nonsupport type of the character: {}".format(self.character_str)
        self.beg_str = "sos"
        self.end_str = "eos"
        if self.loss_type == "attention":
            dict_character = [self.beg_str, self.end_str] + dict_character
        elif self.loss_type == "srn":
            dict_character = dict_character + [self.beg_str, self.end_str]
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if self.character_type == "en":
            text = text.lower()

        text_list = []
        for char in text:
            if char not in self.dict:
                continue
            text_list.append(self.dict[char])
        text = np.array(text_list)
        return text

    def decode(self, text_index, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        char_list = []
        char_num = self.get_char_num()

        if self.loss_type == "attention":
            beg_idx = self.get_beg_end_flag_idx("beg")
            end_idx = self.get_beg_end_flag_idx("end")
            ignored_tokens = [beg_idx, end_idx]
        else:
            ignored_tokens = [char_num]

        for idx in range(len(text_index)):
            if text_index[idx] in ignored_tokens:
                continue
            if is_remove_duplicate:
                if idx > 0 and text_index[idx - 1] == text_index[idx]:
                    continue
            char_list.append(self.character[int(text_index[idx])])
        text = ''.join(char_list)
        return text

    def get_char_num(self):
        return len(self.character)

    def get_beg_end_flag_idx(self, beg_or_end):
        if self.loss_type == "attention":
            if beg_or_end == "beg":
                idx = np.array(self.dict[self.beg_str])
            elif beg_or_end == "end":
                idx = np.array(self.dict[self.end_str])
            else:
                assert False, "Unsupport type %s in get_beg_end_flag_idx"\
                    % beg_or_end
            return idx
        else:
            err = "error in get_beg_end_flag_idx when using the loss %s"\
                % (self.loss_type)
            assert False, err
