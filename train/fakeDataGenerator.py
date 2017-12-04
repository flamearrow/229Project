for i in range(22, 24):
  for f in "PMF":
    for index in range(1, 101):
      open(f + "_S0_M" + str(i) + "_" + str(index) + ".jpg", 'w').close()


