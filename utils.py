def update_mean(value, mean, count):
  """
  Update value of a streaming mean.
  :param value:     New value.
  :param mean:      Mean value.
  :param count:     Number of values averaged.
  :return:
  """

  return (value - mean) / (count + 1)