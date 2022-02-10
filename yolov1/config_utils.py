def parse_config(config_file_path):
  """
  Parse the config gile into a list of dicts
  Each dict contains parameter of a block
  """

  config = []

  with open(config_file_path, 'r') as file:

    for line in file:
      line = line.strip()

      # Skip the empty or commented line
      if line == "" or line.startswith('#'):
        continue

      # New block start with "["
      if line.startswith('['):
        block_name = line[1:-1]
        block = {}
        config.append((block_name, block))
      # Parameters of block
      else:
        key, values = [s.strip() for s in line.split('=')]
        values = [s.strip() for s in values.split(',')]

        # Parse the value of parameters
        for i, value in enumerate(values):
          try:
            value = int(value)
          except ValueError:
            try:
              value = float(value)
            except:
              # value is string
              pass
          values[i] = value

        if len(values) == 1:
          values = values[0]

        block[key] = values
  return config