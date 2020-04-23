import sys

from random import randint

def main():

  n = int(sys.argv[1])

  with open("distance_matrix.txt", 'w+') as f:

    for i in range(n*n):

      if i%n==0:

        f.write("0 ")

      else:

        f.write("%d " % (randint(1, 1000)))


main()
