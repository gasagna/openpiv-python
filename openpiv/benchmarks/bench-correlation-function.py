import openpiv
import os
import time


datadir = '/home/davide/Projects/openpiv-python/openpiv/docs/sphinx-docs/downloads/tutorial-part1/'

imp = openpiv.ImagePair( os.path.join(datadir, 'exp1_001_a.bmp'),
                         os.path.join(datadir, 'exp1_001_b.bmp'), index=0 )
                         
                         
pp = openpiv.ProcessParameters()
pp.window_size = 32
pp.overlap = 16

pp.pretty_print()

N = 30
t0 = time.time()
for i in range(N):
    ff = imp.process(pp)
tf = time.time()

print "Took %.3f seconds for each process" % ((tf-t0)/N)


#cosi mi fa 0.393 per ogni step







