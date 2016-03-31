from sim3 import run

sum = 0
for i in xrange(1000):
    a, b = run()
    diff = a - b
    sum += diff
    print i, sum / (i + 1)

print sum / 1000