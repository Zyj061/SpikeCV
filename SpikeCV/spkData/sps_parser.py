class SPSParser:
    def __init__(self, name):
        self.name = name

    def get_line_num_and_channel(self, data):
      last_2_bytes = (data[-1] << 8) | data[-2]
      c = int((last_2_bytes & 0xC000) >> 14)
      l = int((last_2_bytes&0x3FF0)>>4)
      return l, c

    def is_one_frame_start(self, data):
      if len(data) != 16:
          return False

      line, channel = self.get_line_num_and_channel(data)
      if line == 500 and channel == 0:
        return True

      return False
    
    def is_one_frame_end(self, data):
      if (len(data) & 0xF) != 0:
          return False

      line, channel = self.get_line_num_and_channel(data)
      if line == 500 and channel == 3:
        return True
      return False

    def is_one_line_end(self, data):
      if (len(data) & 0xF) != 0:
          return False

      line, channel = self.get_line_num_and_channel(data)      
      if line in range(1, 501) and channel in range(0, 4):
        return True

      return False
